from datetime import timedelta

import pandas as pd

from carbon_sdk_client.openapi_client.api.carbon_aware_api import CarbonAwareApi
from carbon_sdk_client.openapi_client.api_client import ApiClient
from carbon_sdk_client.openapi_client.configuration import Configuration
from carbon_sdk_client.openapi_client.models import EmissionsForecastDTO, EmissionsForecastBatchParametersDTO


LOCATIONS = ['westcentralus', 'ukwest', 'uksouth', 'westeurope', 'westus', 'australiacentral', 'australiaeast', 'swedencentral', 'norwaywest', 'norwayeast', 'northeurope', 'centralus', 'francesouth', 'francecentral']

FORECAST_FREQ = 15  # in min
FORECAST_WINDOW = 6 * 60  # in min
BACKTEST_START = "2022-10-12T00:00:00.00"
BACKTEST_END = "2022-10-17T00:55:00.00"


def backtest_daterange():
    return pd.date_range(start=BACKTEST_START, end=BACKTEST_END, freq=f"{FORECAST_FREQ}min")


def historic_actual(api_instance, location, start_time, end_time) -> pd.DataFrame:
    api_response = api_instance.get_emissions_data_for_location_by_time(location, time=start_time, to_time=end_time)
    result = pd.DataFrame([r.to_dict() for r in api_response]).drop(columns={"duration"}).sort_values("time")
    result["time"] = result["time"].dt.tz_localize(None)  # remove timezone
    result["location"] = location  # api returns watttime label instead of input label
    result = result.set_index(["location", "time"])
    return result


def historic_forecast(api_instance, locations, query_time) -> pd.DataFrame:
    emissions_forecast_batch_parameters_dtos = [
        EmissionsForecastBatchParametersDTO(
            requested_at=query_time,
            location=location,
            data_start_at=query_time + timedelta(minutes=FORECAST_FREQ),
            data_end_at=query_time + timedelta(minutes=FORECAST_WINDOW + FORECAST_FREQ),
            window_size=5,
        )
    for location in locations]

    api_response = api_instance.batch_forecast_data_async(
        emissions_forecast_batch_parameters_dto=emissions_forecast_batch_parameters_dtos, async_req=False)

    results = []
    for i_entry, entry in enumerate(api_response, 0):
        result = pd.DataFrame(entry.to_dict()['forecast_data']).drop(columns={"duration"}).sort_values("timestamp")
        result = result.rename(columns={'timestamp': 'time', 'value': 'rating'})
        result["time"] = result["time"].dt.tz_localize(None)  # remove timezone
        result = result[result["time"] >= query_time]  # api sometimes returns older values than start_time
        result["location"] = locations[i_entry]  # api returns watttime label instead of input label
        result["query_time"] = query_time
        result = result.set_index(["location", "query_time", "time"])
        results.append(result)

    return pd.concat(results)


def main():
    with ApiClient(configuration=Configuration(host="https://carbon-aware-api.azurewebsites.net")) as api_client:
        api_instance = CarbonAwareApi(api_client)

        # crawl historical actuals
        dfs = []
        for location in LOCATIONS:
            for dt in pd.date_range(start=BACKTEST_START, end=BACKTEST_END, freq=f"1d"):
                print(f"Query {dt} for {location}")
                dfs.append(historic_actual(api_instance, location, dt, dt + timedelta(days=1)))
        df = pd.concat(dfs)
        df = df.groupby(level=df.index.names).first()  # remove duplicates
        df = df[df.index.get_level_values(1).minute.isin([0, 15, 30, 45])]  # downsample to 15min
        df.to_csv("backtest/actuals.csv")

        # crawl historical forecasts
        dfs = []
        for dt in backtest_daterange():
            print(f"Query historical forecast at {dt}")
            dfs.append(historic_forecast(api_instance, LOCATIONS, dt))
        df = pd.concat(dfs)
        # TODO
        df = df[df.index.get_level_values(2).minute.isin([0, 15, 30, 45])]  # downsample to 15min
        df.to_csv("backtest/forecasts.csv")


if __name__ == "__main__":
    main()
