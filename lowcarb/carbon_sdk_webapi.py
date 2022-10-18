from datetime import datetime, timedelta
from typing import List
import requests

from pandas import DataFrame
import pandas as pd

from scipy.interpolate import interp1d

from dateutil import parser


class CarbonSDK_WebAPI():
    strftime = '%Y-%m-%dT%H:%M:%S'

    def __init__(self, url='https://carbon-aware-api.azurewebsites.net'):
        self.url = url

    def get_forecast(self, region: str, windowSize: int) -> DataFrame:
        '''
        Fetches forecast from the carbon_sdk and merges it into a pandas Dataframe


        :param region: string of the region -> Free version of WhattTime only allows 'westis'
        :param windowSize: expected time of the workload in minutes
        :return: pandas Dataframe with 'time' and 'carbon value' column
        :raises: raises InvalidSchemaif anything goes wrong
        '''
        response = requests.get(f'{self.url}/emissions/forecasts/current?location={region}&windowSize={windowSize}')
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

    def get_forecast_batch(self, regions: List[str], windowSize: int) -> DataFrame:
        '''
        Fetches forecast from carbon_sdk and merges it into a pandas Dataframe for all regions
        :param regions: list of strings of regions
        :param windowSize: expected time of the workload in minutes
        :return: pandas Dataframe with 'time', 'carbon value' and 'region' column
        :raises: raises InvalidSchema if anything goes wrong
        '''
        dfs = []
        for region in regions:
            df_region = self.get_forecast(region=region, windowSize=windowSize)
            dfs.append(df_region)

        total_df = pd.concat(dfs)
        return total_df

    def get_history(self, region: str, start_time: datetime, end_time: datetime) -> DataFrame:
        '''
        Fetches the history carbon values for the region from start_time until end_time and merges it into a pandas Dataframe
        :param region:
        :param start_time:
        :param end_time:
        :return: pandas Dataframe with 'time', 'carbon value' and 'region' column
        :raises: raises InvalidSchema if anything goes wrong
        '''

        start_time_string = start_time.strftime(self.strftime)
        end_time_string = end_time.strftime(self.strftime)

        response = requests.get(f'{self.url}/emissions/bylocation?location={region}&time={start_time_string}&toTime={end_time_string}')

        if response.status_code == 200:
            response_json = response.json()
            df = pd.DataFrame({
                'time': [parser.parse(entry['time']) for entry in response_json],
                'value': [entry['rating'] for entry in response_json],
                'region': region
                })


            df = df.pipe(timestamp)

            return df
        else:
            raise requests.exceptions.InvalidSchema(response.status_code)

    def get_history_batch(self, regions: List[str], start_time: datetime, end_time: datetime) -> DataFrame:
        '''
        Fetches the history carbon values for all regions from start_time until end_time and merges it into a pandas Dataframe
        :param region:
        :param start_time:
        :param end_time:
        :return: pandas Dataframe with 'time', 'carbon value' and 'region' column
        :raises: raises InvalidSchema if anything goes wrong
        '''
        dfs = []
        for region in regions:
            df_region = self.get_history(region=region, start_time=start_time, end_time=end_time)
            dfs.append(df_region)

        total_df = pd.concat(dfs)
        return total_df

    def get_carbon_average(self, region: str, start_time: datetime, end_time: datetime) -> float:
        '''
        Fetches the carbon average for the region from start_time to end_time and
        :param region:
        :param start_time:
        :param end_time:
        :return: carbon average
        '''
        start_time_string = start_time.strftime(self.strftime)
        end_time_string = end_time.strftime(self.strftime)

        response = requests.get(f'{self.url}/emissions/average-carbon-intensity?location={region}&startTime={start_time_string}&endTime={end_time_string}')

        if response.status_code == 200:
            return response.json()['carbonIntensity']
        else:
            raise requests.exceptions.InvalidSchema(response.status_code)

    def get_carbon_average_batch(self, regions: List[str],start_time: datetime, end_time: datetime) -> DataFrame:
        '''
        Fetches the carbon average for all regions from start_time to end_time and puts it into a pandas DataFrame
        :param regions:
        :param start_time:
        :param end_time:
        :return: pandas dataframe with 'region' and 'average_value'
        '''
        average_values = [self.get_carbon_average(region, start_time=start_time, end_time=end_time) for region in regions]

        df = pd.DataFrame({
            'region': regions,
            'average_value': average_values
        })

        return df

    def vs_carbon_average(self, df: DataFrame, start_time: datetime, end_time:datetime) -> DataFrame:
        '''
        Pandas pipe function that takes the carbon 'value' column and normalizes it against the carbon average (between start_time and end_time) from the 'region' into 'value_vs_average'
        :param df: dataframe with 'value' and 'region' column
        :param start_time:
        :param end_time:
        :return: input dataframe with additional 'value_vs_average' column
        '''
        regions = df['region'].unique()

        average_values = self.get_carbon_average_batch(regions, start_time=start_time, end_time=end_time)

        df = df.merge(average_values, on='region')
        df['value_vs_average'] = df['value'] / df['average_value']

        return df

    def get_historic_forecast(self, region: str, start_time: datetime, windowSize: int, roundtime: int) -> DataFrame:
        requested_time_string = (start_time - timedelta(minutes=5)).strftime(self.strftime)
        start_time_string = start_time.strftime(self.strftime)
        end_time_string = (start_time + timedelta(hours=roundtime)).strftime(self.strftime)


        query_data = [
            {"requestedAt": requested_time_string,
             "location": region,
             "dataStartAt": start_time_string,
             "dataEndAt": end_time_string,
             "windowSize": windowSize}
        ]
        response = requests.post('https://carbon-aware-api.azurewebsites.net/emissions/forecasts/batch', json=query_data)
        entrys = response.json()[0]['forecastData']

        df_forecast = pd.DataFrame({
            'region': region,
            'time': [parser.parse(entry['timestamp']) for entry in entrys],
            'value': [entry['value'] for entry in entrys]
        })

        df_forecast = df_forecast.pipe(timestamp)

        return df_forecast


    def get_historic_forecast_batch(self, regions: List[str], start_time: datetime, windowSize: int, roundtime: int) -> DataFrame:
        dfs = []
        for region in regions:
            df_region = self.get_historic_forecast(region=region, start_time=start_time, windowSize=windowSize, roundtime=roundtime)
            dfs.append(df_region)

        total_df = pd.concat(dfs)
        return total_df

def timestamp(df: DataFrame) -> DataFrame:
    df = df.sort_values(by='time').reset_index(drop=True)
    df['timestamp'] = df['time'].map(pd.Timestamp.timestamp)
    df['timestamp_indv'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 3600
    return df