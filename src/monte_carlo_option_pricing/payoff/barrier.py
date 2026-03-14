"""
Barrier option payoff implementations.

Barrier options are path-dependent derivatives whose payoff depends
on whether the underlying asset price crosses a specified barrier
level during the life of the option.

This module implements knock-out barrier options. A knock-out option
becomes worthless if the barrier is breached at any point during the
simulation path.
"""

from __future__ import annotations

import numpy as np

from .base import Payoff, register_payoff


class UpAndOutCall(Payoff):
    """
    Up-and-out call option payoff.

    A European call option that becomes worthless if the asset price
    ever rises above the barrier level during the option's life.

    Payoff:

        max(S_T - K, 0)    if max(S_t) <= barrier
        0                  otherwise

    Parameters
    ----------
    K : float
        Strike price.
    barrier : float
        Upper barrier level.
    """

    name: str = "up_and_out_call"

    def __init__(self, K: float, barrier: float):
        super().__init__(K)
        self.barrier = barrier

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
            Payoff values for each path.
        """
        terminal_price = S[:, -1]
        payoff = np.maximum(terminal_price - self.K, 0.0)

        knocked_out = np.max(S, axis=1) > self.barrier
        payoff[knocked_out] = 0.0

        return payoff


class DownAndOutPut(Payoff):
    """
    Down-and-out put option payoff.

    A European put option that becomes worthless if the asset price
    ever falls below the barrier level during the option's life.

    Payoff:

        max(K - S_T, 0)    if min(S_t) >= barrier
        0                  otherwise

    Parameters
    ----------
    K : float
        Strike price.
    barrier : float
        Lower barrier level.
    """

    name: str = "down_and_out_put"

    def __init__(self, K: float, barrier: float):
        super().__init__(K)
        self.barrier = barrier

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
            Payoff values for each path.
        """
        terminal_price = S[:, -1]
        payoff = np.maximum(self.K - terminal_price, 0.0)

        knocked_out = np.min(S, axis=1) < self.barrier
        payoff[knocked_out] = 0.0

        return payoff


register_payoff(UpAndOutCall)
register_payoff(DownAndOutPut)