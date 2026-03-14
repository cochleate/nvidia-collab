import cupy as cp
from src.interfaces import MonteCarloOptionPricer

class CuPyMonteCarlo(MonteCarloOptionPricer):
    def __init__(self, S0, K, T, r, sigma, payoff_fn=None):
        super().__init__(S0, K, T, r, sigma)
        self.payoff_fn = payoff_fn or (lambda S: (S[:,-1] - self.K).clip(min=0))

    def __call__(self, paths=1_000_000, steps=50):
        dt = self.T / steps
        S = cp.full((paths, steps), self.S0, dtype=cp.float32)

        for t in range(1, steps):
            Z = cp.random.standard_normal(paths, dtype=cp.float32)
            S[:,t] = S[:,t-1] * cp.exp((self.r - 0.5*self.sigma**2)*dt + self.sigma*cp.sqrt(dt)*Z)

        payoff = self.payoff_fn(S)
        return float(cp.exp(-self.r * self.T) * cp.mean(payoff))