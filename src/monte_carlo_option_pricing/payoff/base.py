from abc import ABC, abstractmethod
import numpy as np


class Payoff(ABC):
    """
    Abstract base class for option payoff functions.
    """

    name: str

    @abstractmethod
    def __call__(self, S: np.ndarray) -> np.ndarray:
        """
        Compute the payoff for simulated asset paths.

        Parameters
        ----------
        S : ndarray
            Simulated asset price paths.

        Returns
        -------
        ndarray
            Payoff values.
        """

# src/payoffs.py
import numpy as np

# --- Vanilla ---
def european_call(S, K):
    return (S[:,-1] - K).clip(min=0)

def european_put(S, K):
    return (K - S[:,-1]).clip(min=0)

# --- Asian ---
def asian_call(S, K):
    return (S.mean(axis=1) - K).clip(min=0)

def asian_put(S, K):
    return (K - S.mean(axis=1)).clip(min=0)

# --- Barrier ---
def up_and_out_call(S, K, barrier=120):
    payoff = (S[:,-1] - K).clip(min=0)
    payoff[np.max(S, axis=1) > barrier] = 0
    return payoff

def down_and_out_put(S, K, barrier=80):
    payoff = (K - S[:,-1]).clip(min=0)
    payoff[np.min(S, axis=1) < barrier] = 0
    return bpayoff

# --- Lookback ---
def floating_strike_call(S):
    return S[:,-1] - np.min(S, axis=1)

def floating_strike_put(S):
    return np.max(S, axis=1) - S[:,-1]

# --- Digital / Binary ---
def cash_or_nothing_call(S, K, payout=1):
    return np.where(S[:,-1] > K, payout, 0)

def cash_or_nothing_put(S, K, payout=1):
    return np.where(S[:,-1] < K, payout, 0)

# Registry mapping strings to functions
PAYOFFS = {
    "european_call": european_call,
    "european_put": european_put,
    "asian_call": asian_call,
    "asian_put": asian_put,
    "up_and_out_call": up_and_out_call,
    "down_and_out_put": down_and_out_put,
    "floating_strike_call": floating_strike_call,
    "floating_strike_put": floating_strike_put,
    "cash_or_nothing_call": cash_or_nothing_call,
    "cash_or_nothing_put": cash_or_nothing_put,
}