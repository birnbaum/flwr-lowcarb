from logging import INFO
from typing import Dict, List, Optional
import random

import pandas as pd
from flwr.common.logger import log

import flwr as fl
from flwr.server.client_manager import ClientManager
from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy

import requests
from pandas import DataFrame
from ray.exceptions import RaySystemError

from backtest.strategy import CarbonAwareStrategy


from lowcarb.carbon_sdk_webapi import CarbonSDK_WebAPI

import numpy as np

class Object(object):
    pass


class LowCarb_ClientManager(fl.server.client_manager.SimpleClientManager):
    '''
    Implementation of lowcarbs ClientManager, which is a simple extension of flower's own standard SimpleClientManager.
    lowcarb achieves carbon aware federated learning by implementing the sample() method to select clients in a way to minmize the carbon foodprint while maintaining net integrity.
    '''
    def __init__(self, api_host, workload_duration=15, forecast_window=12, **kargs):
        super(LowCarb_ClientManager, self).__init__(**kargs)
        self.host = api_host
        self.api = CarbonSDK_WebAPI(self.host)
        self.workload_duration = workload_duration
        self.forecast_window = forecast_window

        self.client_participation: dict[str: int] = {}

    def sample(
            self,
            num_clients: int,
            min_num_clients: Optional[int] = None,
            criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:




        #####each sample round add any new client to the participitation DataFrame
        for cid in list(self.clients.keys()):
            if not (cid in self.client_participation.keys()):
                self.client_participation[cid] = 0


        """Sample a number of Flower ClientProxy instances."""
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

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # sampled_cids = random.sample(available_cids, num_clients)



        # data for the strategy to select the next best clients
        client_locations = self._get_client_locations()
        present_locations = pd.Series(client_locations.values()).unique()
        forecasts = self._get_location_forecasts(present_locations)

        available_cids_participation = {cid: self.client_participation[cid] for cid in available_cids}

        strategy = CarbonAwareStrategy(clients_per_round=num_clients, max_forecast_duration=self.forecast_window)
        selected_clients = strategy.select(forecasts=forecasts, past_participation=available_cids_participation, client_location_map=client_locations)

        for client in selected_clients:
            self.client_participation[client] = self.client_participation[client] + 1


        print('_______________________________________________________________________\n Available Clients with their locations\n_______________________________________________________________________')
        for client, location in client_locations.items():
            print(f'{client} {location}')

        print('_______________________________________________________________________\n Available Clients with their participation\n_______________________________________________________________________')
        for client, participation in available_cids_participation.items():
            print(f'{client} {participation}')

        print('_______________________________________________________________________\n selected low carbon clients\n_______________________________________________________________________')
        for client in selected_clients:
            print(f'{client}')

        return [self.clients[cid] for cid in selected_clients]

        # return [self.clients[cid] for cid in sampled_cids]

    def _get_client_locations(self) -> dict[str, str]:
        '''
        Fetches all the clients' location and puts it in a DataFrame
        :return: pandas Dataframe with 'cid' and 'location' column
        '''
        client_props = []
        for cid, client in self.clients.items():
            Ins = Object()  ##### weird hack for gRPC, I guess the Ins for get_properties are missing in flwr
            Ins.config = {'config_value': 'config_sample_value'}
            try:
                client_prop = client.get_properties(ins=Ins, timeout=500)
                client_prop.properties.update({'cid': cid})
                client_props.append(client_prop.properties)
            except RaySystemError as error:
                print(error)


        location_df = pd.DataFrame({
            'cid': [client_prop['cid'] for client_prop in client_props],
            'location': [client_prop['location'] for client_prop in client_props]
        })

        client_locations = {client['cid']: client['location'] for i, client in
                                location_df.iterrows()}

        return client_locations

    def _get_location_forecasts(self, locations) -> Dict[str, list]:
        forecasts_response = self.api.get_forecast_batch(locations, windowSize=self.workload_duration, forecast_window=self.forecast_window)
        forecasts = {region: forecast['value'].to_list() for region, forecast in forecasts_response.groupby('region')}
        return forecasts