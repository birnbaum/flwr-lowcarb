import pandas as pd

from strategy import Strategy, RandomStrategy, CarbonAwareStrategy
from crawler import backtest_daterange, LOCATIONS

# TODO: For now we assume one client per location


def run_backtest(actuals_path: str, forecasts_path: str, strategy: Strategy):
    with open(actuals_path, "r") as f:
        actuals = pd.read_csv(f, index_col=[0, 1], parse_dates=True)["rating"]
    with open(forecasts_path, "r") as f:
        forecasts = pd.read_csv(f, index_col=[0, 1, 2], parse_dates=True)["rating"]

    participation = {location: 0 for location in LOCATIONS}  # counts number of times a location was selected
    emissions = {location: 0 for location in LOCATIONS}  # counts emissions per location
    rounds = 0
    for dt in backtest_daterange():
        df = forecasts[forecasts.index.get_level_values(1) == dt]  # level 1 is query time
        df.index = df.index.droplevel(1)
        if df.empty:
            break

        location_forecasts = {location: None for location in LOCATIONS}  # important for sort order
        for location, location_forecast in df.groupby(level=0):
            location_forecast.index = location_forecast.index.droplevel(0)  # dropping location from index
            location_forecast[dt] = actuals.loc[(location, dt)]
            location_forecasts[location] = location_forecast.sort_index()

        selected_locations = strategy.select(location_forecasts, participation)
        for location in selected_locations:
            participation[location] += 1
            emissions[location] += actuals.loc[(location, dt)]
        rounds += 1

    return participation, emissions, rounds


if __name__ == "__main__":
    # run_backtest("backtest/actuals.csv", "backtest/forecasts.csv", strategy=RandomStrategy(3))
    run_backtest("backtest/actuals.csv", "backtest/forecasts.csv", strategy=CarbonAwareStrategy(clients_per_round=3, max_forecast_duration=36))