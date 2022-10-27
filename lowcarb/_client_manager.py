from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from flwr.common import GetPropertiesIns
from flwr.server import SimpleClientManager
from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy

from lowcarb._strategy import CarbonAwareStrategy
from lowcarb.python_carbon_sdk_client.openapi_client.configuration import Configuration
from lowcarb.python_carbon_sdk_client.openapi_client.api_client import ApiClient
from lowcarb.python_carbon_sdk_client.openapi_client.api.carbon_aware_api import CarbonAwareApi


class LowcarbClientManager(SimpleClientManager):
    """Extends Flower's SimpleClientManager by selecting clients in a carbon-aware manner instead of randomly."""

    def __init__(self, api_host, workload_duration=15, forecast_window=12):
        self.host = api_host
        self.api_client = ApiClient(configuration=Configuration(host=api_host))
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
        client_location_map = self._get_client_locations()
        carbon_forecasts = self._get_carbon_forecasts(set(client_location_map.values()))
        available_cids_participation = {cid: self.client_participation[cid] for cid in available_cids}

        strategy = CarbonAwareStrategy(num_clients=num_clients, max_forecast_duration=self.forecast_window)
        sampled_cids = strategy.select(forecasts=carbon_forecasts, past_participation=available_cids_participation,
                                       client_location_map=client_location_map)

        for client in sampled_cids:
            self.client_participation[client] += 1

        return sampled_cids

    def _get_client_locations(self) -> Dict[str, str]:
        """Fetches the clients' locations."""
        return {cid: client.get_properties(ins=GetPropertiesIns(config={}), timeout=500).properties["location"]
                for cid, client
                in self.clients.items()}

    def _get_carbon_forecasts(self, locations) -> Dict[str, list]:
        """Returns the carbon forecast for each provided location."""
        dfs = []
        for location in locations:
            end = datetime.now() + timedelta(hours=self.forecast_window)
            response = self.api_instance.get_current_forecast_data([location],
                                                                   data_end_at=end,
                                                                   window_size=self.forecast_window)
            dfs.append(pd.DataFrame({
                'time': [entry['timestamp'] for entry in response[0]['forecast_data']],
                'value': [entry['value'] for entry in response[0]['forecast_data']],
                'location': locations
            }))
        df = pd.concat(dfs)
        forecasts = {region: forecast['value'].to_list() for region, forecast in df.groupby('region')}
        return forecasts
