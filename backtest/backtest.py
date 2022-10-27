import numpy as np
import pandas as pd

from crawler import backtest_daterange, LOCATIONS
from lowcarb._strategy import Strategy, RandomStrategy, CarbonAwareStrategy


def run_backtest(actuals_path: str, forecasts_path: str, n_clients: int, strategy: Strategy):
    with open(actuals_path, "r") as f:
        actuals = pd.read_csv(f, index_col=[0, 1], parse_dates=True)["rating"]
    with open(forecasts_path, "r") as f:
        forecasts = pd.read_csv(f, index_col=[0, 1, 2], parse_dates=True)["rating"]

    rng = np.random.default_rng(0)
    client_location_map = {c: rng.choice(LOCATIONS) for c in range(n_clients)}

    participation = {c: 0 for c in range(n_clients)}  # counts number of times a location was selected
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

        selected_clients = strategy.select(forecasts={l: fc.values for l, fc in location_forecasts.items()},
                                           past_participation=participation,
                                           client_location_map=client_location_map)
        for c in selected_clients:
            participation[c] += 1
            location = client_location_map[c]
            emissions[location] += actuals.loc[(location, dt)]
        rounds += 1

    return participation, emissions, rounds


if __name__ == "__main__":
    # RandomStrategy(num_clients=10)
    strategy = CarbonAwareStrategy(num_clients=10, max_forecast_duration=36)
    run_backtest("backtest/actuals.csv", "backtest/forecasts.csv", n_clients=100, strategy=strategy)
