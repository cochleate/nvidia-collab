"""
Abstract interface for Monte Carlo option pricers.

Concrete implementations may execute on CPU or GPU backends but must
expose a callable interface for computing option prices from simulated
asset paths.
"""

from abc import ABC, abstractmethod


class MonteCarloOptionPricer(ABC):
    """
    Base class for Monte Carlo option pricing implementations.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    """

    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float) -> None:
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    @abstractmethod
    def __call__(self, paths: int, steps: int) -> float:
        """
        Compute the Monte Carlo estimate of the option price.

        Parameters
        ----------
        paths : int
            Number of simulated Monte Carlo paths.
        steps : int
            Number of time steps in each path.

        Returns
        -------
        float
            Discounted expected payoff of the option.
        """