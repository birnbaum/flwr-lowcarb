from collections import defaultdict
from typing import Dict, List, Optional

from flwr.server import SimpleClientManager
from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy

from backtest.strategy import CarbonAwareStrategy
from lowcarb.carbon_sdk_webapi import CarbonSDK_WebAPI


class Object:
    pass


class LowcarbClientManager(SimpleClientManager):
    """Extends Flower's SimpleClientManager by selecting clients in a carbon-aware manner instead of randomly."""

    def __init__(self, api_host, workload_duration=15, forecast_window=12):
        self.host = api_host
        self.api = CarbonSDK_WebAPI(self.host)
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
            # TODO Not sure if I get this, couldn't this be a dict
            #  {'config': {'config_value': 'config_sample_value'}} ?
            Ins = Object()  # weird hack for gRPC, I guess the Ins for get_properties are missing in flwr
            Ins.config = {'config_value': 'config_sample_value'}
            client_prop = client.get_properties(ins=Ins, timeout=500)
            client_prop.properties.update({'cid': cid})
            client_props.append(client_prop.properties)
        return {client_prop['cid']: client_prop['location'] for client_prop in client_props}

    def _get_location_forecasts(self, locations) -> Dict[str, list]:
        # TODO use OpenAPI client
        forecasts_response = self.api.get_forecast_batch(locations, windowSize=self.workload_duration,
                                                         forecast_window=self.forecast_window)
        forecasts = {region: forecast['value'].to_list() for region, forecast in forecasts_response.groupby('region')}
        return forecasts
