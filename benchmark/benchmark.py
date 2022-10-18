from datetime import timedelta

import numpy as np
import pandas as pd
from IPython import display
from more_itertools import windowed
from pandas import DataFrame
from scipy.integrate import trapz

from lowcarb.carbon_sdk_webapi import CarbonSDK_WebAPI


def rolling_integration(df: DataFrame, origin: str, target: str, windowSize: int) -> DataFrame:
    time_step = (np.gradient(df['timestamp']) / 3600).mean()
    rolling_window = int(windowSize / (time_step * 60))  ##depends on the windowSize and the workload length

    def rolling_trapz(df: DataFrame) -> DataFrame:
        return trapz(y=df, dx=time_step)

    df[target] = (df[origin].iloc[::-1].rolling(rolling_window).apply(rolling_trapz).iloc[::-1]).reset_index(drop=True)

    return df


def run_benchmark(config, client_selection):
    api = CarbonSDK_WebAPI('https://carbon-aware-api.azurewebsites.net')

    selected_regions = config['selected_regions']
    sim_start = config['sim_start']
    sim_end = config['sim_end']
    round_time = config['round_time']
    windowSize = config['windowSize']
    num_clients = config['num_clients']

    round_timedelta = timedelta(hours=round_time)
    n_rounds = int((sim_end - sim_start) / round_timedelta)
    round_start_times = [sim_start + (i * round_timedelta) for i in range(n_rounds + 1)]

    clients_df = pd.DataFrame({
        'region': selected_regions,
        'trained': 0,
        'untrained_since': 0,
        'emissions_total': 0
    })

    for i_round, times in enumerate(windowed(round_start_times, 2), 1):
        round_start_time, round_end_time = times


        ###select clients using the supplied function
        selected_clients = client_selection(config, i_round, round_start_time, clients_df, num_clients)

        ###fetch carbon_data for each selected client
        carbon_history = api\
            .get_history_batch(regions=clients_df.loc[selected_clients.index, 'region'], start_time=round_start_time, end_time=round_end_time)\
            .groupby('region',group_keys=False)\
            .apply(rolling_integration, origin='value', target='rolling_int', windowSize=windowSize)


        #this is a total mess and needs to be cleaned up at some point
        emissions_df = pd.DataFrame(
            [tuple(carbon_history[(carbon_history['region'] == client['region']) & (carbon_history['timestamp_indv'] == client['scheduled_time'])][['region', 'rolling_int']].iloc[0])
             for _, client in selected_clients.iterrows()],
            columns=['region', f'emissions_{i_round}']
        )

        clients_df = clients_df.merge(emissions_df, on='region', how='outer')
        clients_df['emissions_total'] = clients_df.iloc[:, 4:].sum(axis=1)

        ###advance client dataframe for one step
        clients_df.loc[selected_clients.index, 'trained'] = clients_df.loc[selected_clients.index, 'trained'] + 1
        clients_df.loc[:, 'untrained_since'] = clients_df.loc[:, 'untrained_since'] + 1
        clients_df.loc[selected_clients.index, 'untrained_since'] = 0

        display.clear_output()
        display.display(clients_df)


    clients_df['emissions_total'] = clients_df.iloc[:, 4:].sum(axis=1)
    display.clear_output()
    display.display(clients_df)

    return clients_df

def run_benchmark_comparison(config):
    api = CarbonSDK_WebAPI('https://carbon-aware-api.azurewebsites.net')

    selected_regions = config['selected_regions']
    sim_start = config['sim_start']
    sim_end = config['sim_end']
    round_time = config['round_time']
    windowSize = config['windowSize']
    num_clients = config['num_clients']

    round_timedelta = timedelta(hours=round_time)
    n_rounds = int((sim_end - sim_start) / round_timedelta)
    round_start_times = [sim_start + (i * round_timedelta) for i in range(n_rounds + 1)]

    benchmark_results = []
    for region in selected_regions:
        carbon_values = []
        for round_start_time in round_start_times:
            region_carbon_history = api.get_history(region, start_time=round_start_time,
                                                    end_time=round_start_time + (
                                                                timedelta(minutes=windowSize) * n_rounds * num_clients))

            time_step = (np.gradient(region_carbon_history['timestamp']) / 3600).mean()

            carbon_value = trapz(y=region_carbon_history['value'], dx=time_step)

            carbon_values.append(carbon_value)
        benchmark_results.append((region, pd.Series(carbon_values).mean()))
        display.clear_output()
        display.display(pd.DataFrame(benchmark_results, columns=['region', 'emission_total']))

    return pd.DataFrame(benchmark_results, columns=['region', 'emission_total'])