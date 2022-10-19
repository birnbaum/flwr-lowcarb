from datetime import timedelta

import pandas as pd

from carbon_sdk_client.openapi_client.api.carbon_aware_api import CarbonAwareApi
from carbon_sdk_client.openapi_client.api_client import ApiClient
from carbon_sdk_client.openapi_client.configuration import Configuration


LOCATIONS = ['westcentralus', 'ukwest', 'uksouth', 'westeurope', 'westus', 'australiacentral', 'australiaeast', 'swedencentral', 'norwaywest', 'norwayeast', 'northeurope', 'centralus', 'francesouth', 'francecentral']
FORECAST_FREQ = 15  # in min
FORECAST_WINDOW = 180  # in min
BACKTEST_START = "2022-10-13T00:00:00.00"
BACKTEST_END = "2022-10-15T00:00:00.00"


def historic_forecast(api_instance, location, start_time, end_time) -> pd.DataFrame:
    api_response = api_instance.get_emissions_data_for_location_by_time(location, time=start_time, to_time=end_time)
    result = pd.DataFrame([r.to_dict() for r in api_response]).drop(columns={"duration"}).sort_values("time")
    result["time"] = result["time"].dt.tz_localize(None)  # remove timezone
    result = result[result["time"] >= start_time]  # api sometimes returns older values than start_time
    result["location"] = location  # api returns watttime label instead of input label
    result["query_time"] = start_time
    result = result.set_index(["location", "query_time", "time"])
    return result


def main():
    with ApiClient(configuration=Configuration(host="https://carbon-aware-api.azurewebsites.net")) as api_client:
        api_instance = CarbonAwareApi(api_client)
        dfs = []
        for location in LOCATIONS:
            for dt in pd.date_range(start=BACKTEST_START, end=BACKTEST_END, freq=f"{FORECAST_FREQ}min"):
                print(f"Query {dt} for {location}")
                dfs.append(historic_forecast(api_instance, location, dt, dt + timedelta(minutes=FORECAST_WINDOW)))
        pd.concat(dfs).to_csv("backtest.csv")


if __name__ == "__main__":
    main()
