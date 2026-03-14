from src.payoffs import PAYOFFS
from src.cpu_monte_carlo import CPUMonteCarlo
from src.gpu_monte_carlo_cupy import CuPyMonteCarlo

def monte_carlo_option_price(S0, K, T, r, sigma, paths, steps,
                             device="cpu", mode="standard", payoff="european_call", **kwargs):
    """
    payoff: string key from PAYOFFS, or a custom function
    kwargs: extra arguments for payoff function (e.g., barrier level, payout)
    """
    # Resolve payoff function
    if isinstance(payoff, str):
        if payoff not in PAYOFFS:
            raise ValueError(f"Unknown payoff: {payoff}")
        payoff_fn = lambda S: PAYOFFS[payoff](S, K, **kwargs)
    else:
        payoff_fn = payoff  # user-provided custom function

    # Select pricer
    device = device.lower()
    if device == "cpu":
        pricer = CPUMonteCarlo(S0, K, T, r, sigma, payoff_fn=payoff_fn)
    elif device == "gpu":
        pricer = CuPyMonteCarlo(S0, K, T, r, sigma, payoff_fn=payoff_fn)
    else:
        raise ValueError(f"Unknown device: {device}")

    return pricer(paths, steps)