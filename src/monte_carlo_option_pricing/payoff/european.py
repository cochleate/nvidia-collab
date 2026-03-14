"""
European option payoff implementations.

This module defines payoff functions for standard European-style
derivative contracts. European options can only be exercised at
maturity and therefore depend solely on the terminal asset price
of each simulated path.
"""

from __future__ import annotations

import numpy as np

from .base import Payoff, register_payoff


class EuropeanCall(Payoff):
    """
    European call option payoff.

    The payoff of a European call option is

        max(S_T - K, 0)

    where S_T is the terminal asset price and K is the strike price.

    Parameters
    ----------
    K : float
        Strike price of the option.
    """

    name: str = "european_call"

    def __call__(self, S: np.ndarray) -> np.ndarray:
        """
        Compute the payoff for simulated asset price paths.

        Parameters
        ----------
        S : ndarray
            Simulated asset price paths with shape (paths, steps).
            The final column represents the terminal price S_T.

        Returns
        -------
        ndarray
            Payoff values for each simulated path.
        """
        terminal_price = S[:, -1]
        return np.maximum(terminal_price - self.K, 0.0)


class EuropeanPut(Payoff):
    """
    European put option payoff.

    The payoff of a European put option is

        max(K - S_T, 0)

    where S_T is the terminal asset price and K is the strike price.

    Parameters
    ----------
    K : float
        Strike price of the option.
    """

    name: str = "european_put"

    def __call__(self, S: np.ndarray) -> np.ndarray:
        """
        Compute the payoff for simulated asset price paths.

        Parameters
        ----------
        S : ndarray
            Simulated asset price paths with shape (paths, steps).
            The final column represents the terminal price S_T.

        Returns
        -------
        ndarray
            Payoff values for each simulated path.
        """
        terminal_price = S[:, -1]
        return np.maximum(self.K - terminal_price, 0.0)


register_payoff(EuropeanCall)
register_payoff(EuropeanPut)