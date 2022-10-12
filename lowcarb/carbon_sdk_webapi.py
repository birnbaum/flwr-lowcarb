from typing import List
import requests

from pandas import DataFrame
import pandas as pd

from dateutil import parser


def get_forecast(region: str, windowSize: int) -> DataFrame:
    '''
    Fetches forecast from the carbon_sdk and merges it into a pandas Dataframe


    :param region: string of the region -> Free version of WhattTime only allows 'westis'
    :param windowSize: expected time of the workload in minutes
    :return: pandas Dataframe with 'time' and 'carbon value' column
    :raises: raises InvalidSchemaif anything goes wrong
    '''
    response = requests.get(f'http://127.0.0.1:5073/emissions/forecasts/current?location={region}&windowSize={windowSize}')
    if response.status_code == 200:
        response_json = response.json()[0]
        df = pd.DataFrame({
            'time': [parser.parse(entry['timestamp']) for entry in response_json['forecastData']],
            'value': [entry['value'] for entry in response_json['forecastData']],
            'region': region
            })
        return df
    else:
        raise requests.exceptions.InvalidSchema(response.status_code)

def get_forecast_batch(regions: List[str], windowSize: int) -> DataFrame:
    '''
    Fetches forecast from carbon_sdk and merges it into a pandas Dataframe for all regions
    :param regions: list of strings of regions
    :param windowSize: expected time of the workload in minutes
    :return: pandas Dataframe with 'time' and 'carbon value' column
    :raises: raises InvalidSchema if anything goes wrong
    '''
    dfs = []
    for region in regions:
        df_region = get_forecast(region=region, windowSize=windowSize)
        dfs.append(df_region)

    total_df = pd.concat(dfs)
    return total_df