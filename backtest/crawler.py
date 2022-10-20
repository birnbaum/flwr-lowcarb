from datetime import timedelta
from dateutil import parser

import pandas as pd

from carbon_sdk_client.openapi_client.api.carbon_aware_api import CarbonAwareApi
from carbon_sdk_client.openapi_client.api_client import ApiClient
from carbon_sdk_client.openapi_client.configuration import Configuration
from carbon_sdk_client.openapi_client.models import EmissionsForecastDTO, EmissionsForecastBatchParametersDTO
from carbon_sdk_client.openapi_client import ApiException


LOCATIONS = ['westcentralus', 'ukwest', 'uksouth', 'westeurope', 'westus', 'australiacentral', 'australiaeast', 'swedencentral', 'norwaywest', 'norwayeast', 'northeurope', 'centralus', 'francesouth', 'francecentral']
FORECAST_FREQ = 15  # in min
FORECAST_WINDOW = 180  # in min
BACKTEST_START = "2022-10-13T00:00:00.00"
BACKTEST_END = "2022-10-15T00:00:00.00"


def backtest_daterange():
    return pd.date_range(start=BACKTEST_START, end=BACKTEST_END, freq=f"{FORECAST_FREQ}min")


def historic_forecast_ideal(api_instance, location, start_time, end_time) -> pd.DataFrame:
    api_response = api_instance.get_emissions_data_for_location_by_time(location, time=start_time, to_time=end_time)
    result = pd.DataFrame([r.to_dict() for r in api_response]).drop(columns={"duration"}).sort_values("time")
    result["time"] = result["time"].dt.tz_localize(None)  # remove timezone
    result = result[result["time"] >= start_time]  # api sometimes returns older values than start_time
    result["location"] = location  # api returns watttime label instead of input label
    result["query_time"] = start_time
    result = result.set_index(["location", "query_time", "time"])
    return result

def historic_forecast(api_instance, location, start_time, end_time) -> pd.DataFrame:
    query_time = parser.parse(start_time) if (type(start_time) == str) else start_time
    start_time = query_time + timedelta(minutes=5)
    end_time = parser.parse(end_time) if (type(end_time) == str) else end_time

    if type(location) == str:
        locations = [location]
    else:
        locations = location

    emissions_forecast_batch_parameters_dtos = [
        EmissionsForecastBatchParametersDTO(
            requested_at=query_time,
            location=location,
            data_start_at=start_time,
            data_end_at=end_time,
            window_size=5,
        )
    for location in locations]

    try:
        api_response = api_instance.batch_forecast_data_async(emissions_forecast_batch_parameters_dto=emissions_forecast_batch_parameters_dtos, async_req=False)

        results = []
        for i_entry, entry in enumerate(api_response, 0):
            result = pd.DataFrame(entry.to_dict()['forecast_data']).drop(columns={"duration"}).sort_values("timestamp")
            result = result.rename(columns={'timestamp':'time', 'value':'rating'})
            result["time"] = result["time"].dt.tz_localize(None)  # remove timezone
            result = result[result["time"] >= start_time]  # api sometimes returns older values than start_time
            result["location"] = locations[i_entry]  # api returns watttime label instead of input label
            result["query_time"] = query_time
            result = result.set_index(["location", "query_time", "time"])

            results.append(result)


        return pd.concat(results)

    except ApiException as e:
        print("Exception when calling CarbonAwareApi->batch_forecast_data_async: %s\n" % e)

def main():
    with ApiClient(configuration=Configuration(host="https://carbon-aware-api.azurewebsites.net")) as api_client:
        api_instance = CarbonAwareApi(api_client)
        dfs = []
        for location in LOCATIONS:
            for dt in backtest_daterange():
                print(f"Query {dt} for {location}")
                dfs.append(historic_forecast(api_instance, location, dt, dt + timedelta(minutes=FORECAST_WINDOW)))
        pd.concat(dfs).to_csv("backtest/history.csv")


if __name__ == "__main__":
    main()
