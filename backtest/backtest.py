from typing import Dict, Callable, Tuple

import numpy as np
import pandas as pd

from crawler import backtest_daterange, LOCATIONS

# TODO: For now we assume one client per location


def run_backtest(history_path: str, strategy: Callable[[Dict[str, pd.DataFrame]], str]):
    with open(history_path, "r") as f:
        history = pd.read_csv(f, index_col=[0, 1, 2], parse_dates=True)

    selections = {location: 0 for location in LOCATIONS}  # counts number of times a location was selected
    emissions = {location: 0 for location in LOCATIONS}  # counts emissions per location
    for dt in backtest_daterange():
        df = history[history.index.get_level_values(1) == dt]  # level 1 is query time
        df.index = df.index.droplevel(1)
        location_forecasts = {}
        for location, location_forecast in df.groupby(level=0):
            location_forecast.index = location_forecast.index.droplevel(0)  # dropping location from index
            location_forecasts[location] = location_forecast

        selected_locations = strategy(location_forecasts)
        for location in selected_locations:
            selections[location] += 1
            emissions[location] += float(location_forecasts[location].loc[dt])

    return selections, emissions


if __name__ == "__main__":
    run_backtest("backtest/history.csv")
