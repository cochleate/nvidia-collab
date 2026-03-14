from numba import cuda, float32
import numpy as np
from src.interfaces import MonteCarloOptionPricer

@cuda.jit
def simulate_paths_kernel(S0, r, sigma, dt, steps, rand_nums, out):
    idx = cuda.grid(1)
    if idx >= out.shape[0]:
        return
    S = S0
    for t in range(steps):
        S = S * math.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rand_nums[idx, t])
    out[idx] = max(S - S0, 0)  # simple European call

class NumbaCudaMonteCarlo(MonteCarloOptionPricer):
    def __init__(self, S0, K, T, r, sigma):
        super().__init__(S0, K, T, r, sigma)

    def __call__(self, paths=1_000_000, steps=50):
        dt = self.T / steps
        rand_nums = np.random.standard_normal((paths, steps)).astype(np.float32)
        out = np.zeros(paths, dtype=np.float32)

        threads_per_block = 256
        blocks = (paths + threads_per_block - 1) // threads_per_block

        simulate_paths_kernel[blocks, threads_per_block](self.S0, self.r, self.sigma, dt, steps, rand_nums, out)
        return np.exp(-self.r * self.T) * np.mean(out)