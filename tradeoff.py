from functools import partial
import math

import numpy as np
import scipy

# Building blocks: support tradeoff curves
def support_tradeoff(alpha, eps, delta):
    left = 1 - delta - (math.exp(eps) * alpha)
    right = math.exp(-1 * eps) * (1 - delta - alpha)

    return max(0, left, right)


def approx_tradeoff(alpha, epsilons, deltas):
    betas = [support_tradeoff(alpha, epsilons[i], deltas[i]) for i in range(0, len(epsilons))]
    return max(betas)


class smdCurveWrapper:
    """
    Wrapper around opendp smdCurves to make them callable
    """

    smd_curve = None

    def __init__(self, smd_curve):
        self.smd_curve = smd_curve

    def __call__(self, delta):
        return self.smd_curve.epsilon(delta)
    
# Given a privacy profile (epsilon(delta) or delta(epsilon)) curve
class tradeoffCurve:

    privacy_profile = None

    deltas = None
    epsilons = None

    def __init__(self, privacy_profile, deltas = None, epsilons = None):
        if deltas is None and epsilons is None:
            raise Exception("Provide either deltas or epsilons")
        
        self.deltas = deltas
        self.epsilons = epsilons
        self.privacy_profile = privacy_profile

    def _setup(self):
        # Compute epsilons or deltas only once
        if self.epsilons is None:
            self.epsilons = [self.privacy_profile(delta) for delta in self.deltas]
        else:
            self.deltas = [self.privacy_profile(eps) for eps in self.epsilons]

    def __call__(self, alpha):
        # Compute epsilons if not available
        if self.epsilons is None or self.deltas is None:
            self._setup()

        # Use max of support tradeoffs to compute beta
        if isinstance(alpha, (list, set, np.ndarray)):
            return [approx_tradeoff(a, self.epsilons, self.deltas) for a in alpha]

        return approx_tradeoff(alpha, self.epsilons, self.deltas)
    

# Analytical tradeoffs
# -----------------------------------------------------------------------------------------


# Gaussian mechanism
# ------------------
def tradeoff_gaussian(alpha, mu):
    return scipy.stats.norm.cdf(scipy.stats.norm.ppf(1 - alpha) - mu)

def get_tradeoff_gaussian(mu):
    return partial(tradeoff_gaussian, mu=mu)

# Analytic version of delta(epsilon) curve for Gaussian Mechanism -> Tudor
def get_gaussian_privacy_profile(sensitivity: float, sigma: float):
    def func(eps: float) -> float:
       return scipy.stats.norm.cdf(sensitivity/(2 * sigma) - eps * sigma / sensitivity) - np.exp(eps) * scipy.stats.norm.cdf(-sensitivity / (2 * sigma) - eps * sigma / sensitivity)
    return func

# Tradeoff function via optimization of the dual -> Tudor
def tradeoff_opt_gaussian(alpha: float, privacy_profile) -> float:
    def _search_function(x: float):
      if x > 0:
        return np.inf
      return privacy_profile(np.log(-x)) - alpha * x

    sol = scipy.optimize.minimize_scalar(_search_function).x
    return 1 -_search_function(sol)


# Laplace mechanism
# -----------------
def tradeoff_laplace(alpha, scale, sensitivity):
    epsilon = sensitivity / scale
    return scipy.stats.laplace.cdf(scipy.stats.laplace.ppf(1 - alpha) - epsilon)

def get_tradeoff_laplace(scale, sensitivity):
    return partial(tradeoff_laplace, scale=scale, sensitivity=sensitivity)

