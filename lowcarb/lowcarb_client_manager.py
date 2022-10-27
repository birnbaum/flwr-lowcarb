from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from flwr.server import SimpleClientManager
from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy
from pandas import DataFrame

from backtest.strategy import CarbonAwareStrategy
from unittest.mock import Mock

from carbon_sdk_client.openapi_client.api.carbon_aware_api import CarbonAwareApi
from carbon_sdk_client.openapi_client.api_client import ApiClient
from carbon_sdk_client.openapi_client.configuration import Configuration
from carbon_sdk_client.openapi_client.exceptions import OpenApiException

class LowcarbClientManager(SimpleClientManager):
    """Extends Flower's SimpleClientManager by selecting clients in a carbon-aware manner instead of randomly."""

    def __init__(self, api_host, workload_duration=15, forecast_window=12):
        self.host = api_host

        self.api_client = ApiClient(configuration=Configuration(host=self.host))
        self.api_instance = CarbonAwareApi(self.api_client)


        self.workload_duration = workload_duration
        self.forecast_window = forecast_window
        self.client_participation: Dict[str: int] = defaultdict(int)
        super().__init__()

    def sample(
            self,
            num_clients: int,
            min_num_clients: Optional[int] = None,
            criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances.

        This implementation is identical to Flower 1.1.0 `SimpleClientManager.sample(...)` besides calling
        `self._sample(available_cids, num_clients)` instead of `random.sample(available_cids, num_clients)`.
        """
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        sampled_cids = self._sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]

    def _sample(self, available_cids, num_clients):
        """Carbon-aware client selection strategy."""
        client_locations = self._get_client_locations()
        present_locations = set(client_locations.values())
        forecasts = self._get_location_forecasts(present_locations)
        available_cids_participation = {cid: self.client_participation[cid] for cid in available_cids}

        strategy = CarbonAwareStrategy(clients_per_round=num_clients, max_forecast_duration=self.forecast_window)
        sampled_cids = strategy.select(forecasts=forecasts, past_participation=available_cids_participation,
                                       client_location_map=client_locations)

        for client in sampled_cids:
            self.client_participation[client] += 1

        print("--- Available Clients with their locations")
        for client, location in client_locations.items():
            print(f'{client} {location}')

        print("--- Available Clients with their participation")
        for client, participation in available_cids_participation.items():
            print(f'{client} {participation}')

        print("--- Selected low carbon clients")
        for client in sampled_cids:
            print(f'{client}')

        return sampled_cids

    def _get_client_locations(self) -> Dict[str, str]:
        """Fetches the clients' locations."""
        client_props = []
        for cid, client in self.clients.items():
            Ins = Mock()
            Ins.config = {}
            client_prop = client.get_properties(ins=Ins, timeout=500)
            client_prop.properties.update({'cid': cid})
            client_props.append(client_prop.properties)
        return {client_prop['cid']: client_prop['location'] for client_prop in client_props}

    def _get_location_forecasts(self, locations) -> Dict[str, list]:
        forecasts_response = self._get_forecast_batch(locations, window_size=self.workload_duration,
                                                         forecast_window=self.forecast_window)
        forecasts = {region: forecast['value'].to_list() for region, forecast in forecasts_response.groupby('region')}
        return forecasts

    def _get_forecast(self, region: str, window_size: int, forecast_window=None) -> DataFrame:
        '''
        Fetches forecast from the carbon_sdk and merges it into a pandas Dataframe

        :param region: string of the region -> Free version of WhattTime only allows 'westis'
        :param window_size: expected time of the workload in minutes
        :return: pandas Dataframe with 'time' and 'carbon value' column
        :raises: raises InvalidSchemaif anything goes wrong
        '''
        try:
            if forecast_window:
                forecast_window = 22 if (forecast_window > 22) else forecast_window
                end = datetime.now() + timedelta(hours=forecast_window)
                response = self.api_instance.get_current_forecast_data([region], data_end_at=end, window_size=window_size)
            else:
                response = self.api_instance.get_current_forecast_data([region], window_size=window_size)

            df = pd.DataFrame({
                'time': [entry['timestamp'] for entry in response[0]['forecast_data']],
                'value': [entry['value'] for entry in response[0]['forecast_data']],
                'region': region
            })
            return df

        except OpenApiException as e:
            print("Exception when calling CarbonAwareApi->get_current_forecast_data: %s\n" % e)



    def _get_forecast_batch(self, regions: List[str], window_size: int, forecast_window = None) -> DataFrame:
        '''
        Fetches forecast from carbon_sdk and merges it into a pandas Dataframe for all regions
        :param regions: list of strings of regions
        :param window_size: expected time of the workload in minutes
        :return: pandas Dataframe with 'time', 'carbon value' and 'region' column
        :raises: raises InvalidSchema if anything goes wrong
        '''
        dfs = []
        for region in regions:
            df_region = self._get_forecast(region=region, window_size=window_size, forecast_window=forecast_window)
            dfs.append(df_region)

        total_df = pd.concat(dfs)
        return total_df