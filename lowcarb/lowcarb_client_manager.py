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

import numpy as np

class Object(object):
    pass


class LowCarb_ClientManager(fl.server.client_manager.SimpleClientManager):
    '''
    Implementation of lowcarbs ClientManager, which is a simple extension of flower's own standard SimpleClientManager.
    lowcarb achieves carbon aware federated learning by implementing the sample() method to select clients in a way to minmize the carbon foodprint while maintaining net integrity.
    '''
    def __init__(self, **kargs):
        super(LowCarb_ClientManager, self).__init__(**kargs)


        self.client_participation = pd.DataFrame({'cid': [], 'participation': []})

    def sample(
            self,
            num_clients: int,
            min_num_clients: Optional[int] = None,
            criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:


        #####each sample round add any new client to the participitation DataFrame
        available_cids = list(self.clients.keys())
        for available_cid in available_cids:
            if not (available_cid in self.client_participation['cid'].to_list()):
                self.client_participation = pd.concat([self.client_participation, pd.DataFrame({'cid': [available_cid], 'participation': [0]})]).reset_index(drop=True)



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
        client_locations = self.get_client_locations()
        present_locations = pd.Series(client_locations.values()).unique()
        forecasts = get_random_forecast(present_locations)
        client_participation = {client['cid']: client['participation'] for i, client in self.client_participation.iterrows()}


        strategy = CarbonAwareStrategy(clients_per_round=num_clients, max_forecast_duration=12)
        selected_clients = strategy.select(forecasts=forecasts, past_participation=client_participation, client_location_map=client_locations)

        self.client_participation.loc[(self.client_participation['cid'].isin(selected_clients), 'participation')] = self.client_participation.loc[(self.client_participation['cid'].isin(selected_clients), 'participation')] + 1


        print('_______________________________________________________________________\n Available Clients with their locations\n_______________________________________________________________________')
        print(client_locations)

        print('_______________________________________________________________________\n Available Clients with their participation\n_______________________________________________________________________')
        print(self.client_participation)

        print('_______________________________________________________________________\n selected low carbon clients\n_______________________________________________________________________')
        print(selected_clients)

        return [self.clients[cid] for cid in selected_clients]

        # return [self.clients[cid] for cid in sampled_cids]

    def get_client_locations(self) -> DataFrame:
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

def get_random_forecast(locations):
    MAX_DURATION_FORECAST = 12
    N_DATAPOINTS = 100
    DATAPOINTS = np.linspace(0, MAX_DURATION_FORECAST, N_DATAPOINTS)
    mesh = np.meshgrid(DATAPOINTS, np.arange(0, len(locations), 1))[0]
    x_offset = np.random.random((len(locations), 1)) * 10
    y_scale = np.random.random((len(locations), 1)) * 250
    slope = np.random.random((len(locations), 1)) * 1
    y_offset = np.random.random((len(locations), 1)) * 500
    wavelength = np.random.random((len(locations), 1))
    noise_strength = np.random.random((len(locations), 1)) * 250
    noise = np.random.random((len(locations), N_DATAPOINTS)) * noise_strength
    bias = np.linspace(0, 1000, len(locations))
    bias = bias.reshape(len(locations), 1)

    sample_forecasts_data = bias + ((np.sin((mesh*wavelength) + x_offset) * y_scale) + ((mesh*slope)+y_offset)) + noise
    sample_forecasts_data = sample_forecasts_data.clip(min=0)

    forecasts = {locations[i]: sample_forecasts_data[i].tolist() for i, data in enumerate(sample_forecasts_data)}
    return forecasts