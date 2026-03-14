import cupy as cp
from src.interfaces import MonteCarloOptionPricer

class MassiveCuPyMonteCarlo(MonteCarloOptionPricer):
    """
    Handles extremely large number of paths in batches to fit GPU memory.
    """
    def __init__(self, S0, K, T, r, sigma, payoff_fn=None, batch_size=10_000_000):
        super().__init__(S0, K, T, r, sigma)
        self.payoff_fn = payoff_fn or (lambda S: (S[:,-1] - self.K).clip(min=0))
        self.batch_size = batch_size

    def __call__(self, paths=100_000_000, steps=50):
        total_payoff = 0
        batches = (paths + self.batch_size - 1) // self.batch_size

        dt = self.T / steps
        for i in range(batches):
            current_batch = min(self.batch_size, paths - i * self.batch_size)
            S = cp.full((current_batch, steps), self.S0, dtype=cp.float32)
            for t in range(1, steps):
                Z = cp.random.standard_normal(current_batch, dtype=cp.float32)
                S[:,t] = S[:,t-1] * cp.exp((self.r - 0.5*self.sigma**2)*dt + self.sigma*cp.sqrt(dt)*Z)
            payoff = self.payoff_fn(S)
            total_payoff += cp.sum(payoff)
        return float(cp.exp(-self.r * self.T) * total_payoff / paths)