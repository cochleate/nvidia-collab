"""
Asian option payoff implementations.

Asian options are path-dependent derivatives whose payoff depends on
the average value of the underlying asset over the life of the option
rather than solely the terminal price.

This module implements arithmetic-average Asian call and put options.
"""

from __future__ import annotations

import numpy as np

from .base import Payoff, register_payoff


class AsianCall(Payoff):
    """
    Arithmetic Asian call option payoff.

    The payoff is defined as

        max(A - K, 0)

    where

        A = (1 / N) * sum_{t=1..N} S_t

    is the arithmetic average of the simulated asset prices across
    the time steps of the path.

    Parameters
    ----------
    K : float
        Strike price of the option.
    """

    name: str = "asian_call"

    def __call__(self, S: np.ndarray) -> np.ndarray:
        """
        Compute the payoff for simulated asset price paths.

        Parameters
        ----------
        S : ndarray
            Simulated asset price paths with shape (paths, steps).

        Returns
        -------
        ndarray
            Payoff values for each simulated path.
        """
        average_price = S.mean(axis=1)
        return np.maximum(average_price - self.K, 0.0)


class AsianPut(Payoff):
    """
    Arithmetic Asian put option payoff.

    The payoff is defined as

        max(K - A, 0)

    where

        A = (1 / N) * sum_{t=1..N} S_t

    is the arithmetic average of the simulated asset prices across
    the time steps of the path.

    Parameters
    ----------
    K : float
        Strike price of the option.
    """

    name: str = "asian_put"

    def __call__(self, S: np.ndarray) -> np.ndarray:
        """
        Compute the payoff for simulated asset price paths.

        Parameters
        ----------
        S : ndarray
            Simulated asset price paths with shape (paths, steps).

        Returns
        -------
        ndarray
            Payoff values for each simulated path.
        """
        average_price = S.mean(axis=1)
        return np.maximum(self.K - average_price, 0.0)


register_payoff(AsianCall)
register_payoff(AsianPut)