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

from lowcarb.carbon_sdk_webapi import get_forecast_batch

class Object(object):
    pass


class LowCarb_ClientManager(fl.server.client_manager.SimpleClientManager):
    '''
    Implementation of lowcarbs ClientManager, which is a simple extension of flower's own standard SimpleClientManager.
    lowcarb achieves carbon aware federated learning by implementing the sample() method to select clients in a way to minmize the carbon foodprint while maintaining net integrity.
    '''
    def __init__(self, **kargs):
        super(LowCarb_ClientManager, self).__init__(**kargs)

    def sample(
            self,
            num_clients: int,
            min_num_clients: Optional[int] = None,
            criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
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




        clients_properties = self.get_client_locations()

        print('_______________________________________________________________________\n Available Clients with their locations\n_______________________________________________________________________')
        print(clients_properties)
        carbon_optimal_cids = self.carbon_aware_sampling(clients_properties)

        print('_______________________________________________________________________\n selected low carbon clients\n_______________________________________________________________________')
        print(carbon_optimal_cids)

        return [self.clients[cid] for cid in carbon_optimal_cids['cid']]

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

        return location_df

    def carbon_aware_sampling(self, available_clients: DataFrame) -> DataFrame:
        '''
        Receives a list of all available clients and their location
        :param available_clients:
        :return: pandas dataframe with the selected carbon aware clients
        '''



        selected_clients = available_clients[available_clients['location'] == 'norway']

        try:
            carbon_forecasts = get_forecast_batch(['westus', 'westus', 'westus', 'westus', 'westus', 'westus'], 10)
        except:
            pass ##hehe jo

        return selected_clients
