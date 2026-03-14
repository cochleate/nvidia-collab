import numpy as np
from src._interface import MonteCarloOptionPricer

class CPUMonteCarlo(MonteCarloOptionPricer):
    def __init__(self, S0, K, T, r, sigma, payoff_fn=None):
        super().__init__(S0, K, T, r, sigma)
        self.payoff_fn = payoff_fn or (lambda S: (S[:,-1] - self.K).clip(min=0))

    def __call__(self, paths=1_000_000, steps=50):
        dt = self.T / steps
        S = np.zeros((paths, steps), dtype=np.float32)
        S[:,0] = self.S0

        for t in range(1, steps):
            Z = np.random.standard_normal(paths).astype(np.float32)
            S[:,t] = S[:,t-1] * np.exp((self.r - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z)

        payoff = self.payoff_fn(S)
        return np.exp(-self.r * self.T) * np.mean(payoff)