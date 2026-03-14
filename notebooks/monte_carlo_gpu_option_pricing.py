# %%
# GPU-Accelerated Monte Carlo Option Pricing
#
# This notebook demonstrates Monte Carlo simulation for pricing European call options using:
# - CPU (NumPy)
# - GPU (CuPy)
# - Massive GPU (10M+ paths, CuPy)
# - Numba CUDA kernel
#
# We will benchmark runtimes, visualize payoff distributions, and show Monte Carlo convergence.

# %%
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import time

from src.monte_carlo_option_pricing import monte_carlo_option_price

# %% [markdown]
# ## 1. Option Parameters

S0 = 100       # Initial stock price
K = 105        # Strike price
T = 1.0        # Time to maturity (years)
r = 0.05       # Risk-free rate
sigma = 0.2    # Volatility
steps = 50     # Number of time steps

# %% [markdown]
# ## 2. Unified API Calls Examples

price_cpu = monte_carlo_option_price(S0, K, T, r, sigma, paths=1_000_000, steps=steps, device="cpu")
price_gpu = monte_carlo_option_price(S0, K, T, r, sigma, paths=1_000_000, steps=steps, device="gpu", mode="standard")
price_massive = monte_carlo_option_price(S0, K, T, r, sigma, paths=10_000_000, steps=steps, device="gpu", mode="massive")
price_numba = monte_carlo_option_price(S0, K, T, r, sigma, paths=1_000_000, steps=steps, device="gpu", mode="numba")

print(f"CPU Price (1M paths): {price_cpu:.4f}")
print(f"GPU CuPy Price (1M paths): {price_gpu:.4f}")
print(f"Massive GPU CuPy Price (10M paths): {price_massive:.4f}")
print(f"Numba CUDA GPU Price (1M paths): {price_numba:.4f}")

# %% [markdown]
# ## 3. Benchmark: Runtime Comparison

implementations = [
    ("CPU (1M paths)", {"device":"cpu", "mode":"standard", "paths":1_000_000}),
    ("GPU CuPy (1M paths)", {"device":"gpu", "mode":"standard", "paths":1_000_000}),
    ("Massive GPU CuPy (10M paths)", {"device":"gpu", "mode":"massive", "paths":10_000_000}),
    ("Numba CUDA GPU (1M paths)", {"device":"gpu", "mode":"numba", "paths":1_000_000})
]

results = {}
for name, args in implementations:
    start = time.time()
    price = monte_carlo_option_price(S0, K, T, r, sigma, steps=steps, **args)
    end = time.time()
    results[name] = {"price": price, "time": end-start}

df = pd.DataFrame(results).T.rename(columns={"price":"Option Price","time":"Runtime (s)"}).sort_values("Runtime (s)")
display(df)

# %% [markdown]
# ## 4. Payoff Distribution (GPU CuPy, 1M paths)

paths_vis = 1_000_000
dt = T / steps
S = cp.full(paths_vis, S0, dtype=cp.float32)

for t in range(steps):
    Z = cp.random.standard_normal(paths_vis, dtype=cp.float32)
    S *= cp.exp((r - 0.5*sigma**2)*dt + sigma*cp.sqrt(dt)*Z)

payoff = cp.maximum(S - K, 0).get()

plt.figure(figsize=(8,5))
plt.hist(payoff, bins=100, color='skyblue', edgecolor='black')
plt.title("Monte Carlo Payoff Distribution (GPU, 1M paths)")
plt.xlabel("Payoff")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# ## 5. Monte Carlo Convergence Plot

path_counts = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
convergence = {"CPU": [], "GPU CuPy": [], "Massive GPU CuPy": []}

for p in path_counts:
    if p <= 1_000_000:
        convergence["CPU"].append(monte_carlo_option_price(S0, K, T, r, sigma, paths=p, steps=steps, device="cpu"))
    else:
        convergence["CPU"].append(np.nan)
    convergence["GPU CuPy"].append(monte_carlo_option_price(S0, K, T, r, sigma, paths=p, steps=steps, device="gpu", mode="standard"))
    convergence["Massive GPU CuPy"].append(monte_carlo_option_price(S0, K, T, r, sigma, paths=p, steps=steps, device="gpu", mode="massive"))

plt.figure(figsize=(8,5))
for label, prices in convergence.items():
    plt.plot(path_counts, prices, marker='o', label=label)
plt.xscale('log')
plt.xlabel("Number of Paths")
plt.ylabel("Option Price")
plt.title("Monte Carlo Convergence: CPU vs GPU")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ## 6. Summary Table & GPU Speedup Visualization

implementations_summary = [
    ("CPU (1M paths)", {"device":"cpu", "mode":"standard", "paths":1_000_000}),
    ("GPU CuPy (1M paths)", {"device":"gpu", "mode":"standard", "paths":1_000_000}),
    ("Massive GPU CuPy (10M paths)", {"device":"gpu", "mode":"massive", "paths":10_000_000}),
    ("Numba CUDA GPU (1M paths)", {"device":"gpu", "mode":"numba", "paths":1_000_000})
]

summary_results = []
for name, args in implementations_summary:
    start = time.time()
    price = monte_carlo_option_price(S0, K, T, r, sigma, steps=steps, **args)
    runtime = time.time() - start
    summary_results.append((name, price, runtime))

df_summary = pd.DataFrame(summary_results, columns=["Implementation", "Option Price", "Runtime (s)"])

cpu_time = df_summary.loc[df_summary["Implementation"]=="CPU (1M paths)","Runtime (s)"].values[0]
df_summary["GPU Speedup"] = df_summary["Runtime (s)"].apply(lambda t: cpu_time/t if t>0 else np.nan)

display(df_summary.sort_values("Runtime (s)"))

plt.figure(figsize=(8,5))
plt.bar(df_summary["Implementation"], df_summary["GPU Speedup"], color="skyblue", edgecolor="black")
plt.ylabel("Speedup vs CPU")
plt.title("GPU Speedup Comparison for Monte Carlo Option Pricing")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")
plt.show()