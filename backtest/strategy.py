from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def select(self,
               forecasts: Dict[str, pd.DataFrame],
               past_participation: Dict[str, int]) -> List[str]:
        """Selects a list of locations"""


class RandomStrategy(Strategy):
    def __init__(self, clients_per_round: int):
        self.clients_per_round = clients_per_round

    def select(self, forecasts, past_participation):
        return np.random.choice(list(forecasts.keys()), size=self.clients_per_round, replace=False)


class CarbonAwareStrategy(Strategy):
    def __init__(self, clients_per_round: int, max_forecast_duration: int):
        self.clients_per_round = clients_per_round
        self.max_forecast_duration = max_forecast_duration

    def select(self, forecasts, past_participation):
        """Optimize for clients with the least absolute potential to improve their carbon intensity soon.

        For each round, we compute an individual forecast window for each client based on its past participation.
        Underparticipating clients get a short window, because we want them to participate sooner.
        Each client gets a score based on its absolute saving potential within the forecast window.
        The  n clients with the least potential get selected.
        """
        participation = np.array(list(past_participation.values()))
        windows = self._calc_forecast_windows(participation)
        forecast_arrays = [df.values for df in forecasts.values()]
        deltas = [self._lowest_delta(fc, w) for fc, w in zip(forecast_arrays, windows)]
        locations_sorted_by_score = pd.Series(forecasts.keys(), index=deltas).sort_index(ascending=False)
        return locations_sorted_by_score.iloc[:self.clients_per_round].values

    def _calc_forecast_windows(self, participation: np.array):
        if participation.max() == 0:
            return np.full(shape=len(participation), fill_value=self.max_forecast_duration)
        # Clients below 50% of max participation have to participate immediately
        # Clients above 50% get windows that linearly scale with the max_forecast_duration
        # Hence, the client with max participation always gets max_forecast_duration
        normalized = np.maximum(0, 2 * participation / participation.max() - 1)
        return np.round(normalized * self.max_forecast_duration).astype(int)

    def _lowest_delta(self, forecast: np.array, window: int):
        """Returns the lowest delta of and forecasted value compared to 'now'"""
        now = forecast[0]
        if window == 0:  # TODO document
            return 100000 - now
        future = forecast[1:window + 1]
        min_delta = (future - now).min()
        return min_delta
