import sys
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def select(self,
               forecasts: Dict[str, List[float]],  # location to forecast list where [0] is now
               past_participation: Dict[str, int],  # client to number of participation rounds
               client_location_map: Dict[str, str]  # maps clients to their locations
               ) -> List[str]:
        """Selects a list of locations"""


class RandomStrategy(Strategy):
    def __init__(self, num_clients: int):
        self.num_clients = num_clients

    def select(self, forecasts, past_participation, client_location_map):
        clients = list(past_participation.keys())
        return np.random.choice(clients, size=self.num_clients, replace=False)


class CarbonAwareStrategy(Strategy):
    def __init__(self, num_clients: int, max_forecast_duration: int):
        self.num_clients = num_clients
        self.max_forecast_duration = max_forecast_duration

    def select(self, forecasts, past_participation, client_location_map):
        """Optimize for clients with the least absolute potential to improve their carbon intensity soon.

        For each round, we compute an individual forecast window for each client based on its past participation.
        Underparticipating clients get a short window, because we want them to participate sooner.
        Each client gets a score based on its absolute saving potential within the forecast window.
        The  n clients with the least potential get selected.
        """
        clients = list(past_participation.keys())
        participations = np.array(list(past_participation.values()))
        windows = self._calc_forecast_windows(participations)
        participation_limit = np.percentile(participations, 80)
        scores = []
        for client, participation, window in zip(clients, participations, windows):
            if participation > participation_limit:
                scores.append(-sys.maxsize)
                continue
            forecast = forecasts[client_location_map[client]]
            scores.append(self._score(forecast=forecast, window=window))
        clients_sorted_by_score = pd.Series(clients, index=scores).sort_index(ascending=False)
        return list(clients_sorted_by_score.iloc[:self.num_clients].values)

    def _calc_forecast_windows(self, participation: np.array):
        """Maps a client's participation to its forecast window.

        Naive approach:
        - Clients below 50% of max participation have to participate immediately.
        - Clients above 50% get windows that linearly scale with the max_forecast_duration.
        - Hence, the client with max participation always gets max_forecast_duration.
        """
        if participation.max() == 0:
            return np.full(shape=len(participation), fill_value=self.max_forecast_duration)
        normalized = np.maximum(0, 2 * participation / participation.max() - 1)
        return np.round(normalized * self.max_forecast_duration).astype(int)

    def _score(self, forecast: list, window: int):
        """Returns the lowest delta of and forecasted value compared to 'now'.

        If the window is 0, we want the client to participate in any case, so we give it a very high score,
        only weighted by the absolute carbon intensity of 'now'.
        """
        forecast = np.array(forecast) if (type(forecast) == list) else forecast
        now = forecast[0]
        if window == 0:
            return sys.maxsize - now
        future = forecast[1:window + 1]
        min_delta = (future - now).min()
        return min_delta
