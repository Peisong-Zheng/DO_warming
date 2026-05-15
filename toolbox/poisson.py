"""Poisson likelihood utilities for binned event-count models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln


def design_matrix(frame: pd.DataFrame, terms: tuple[str, ...]) -> np.ndarray:
    """Return a numeric design matrix for a tuple of predictor columns."""

    if not terms:
        return np.empty((len(frame), 0), dtype=float)
    return frame.loc[:, list(terms)].to_numpy(dtype=float)


def binned_poisson_loglik(beta: np.ndarray, x: np.ndarray, y: np.ndarray, dt: np.ndarray) -> float:
    """Poisson log likelihood for binned counts with a log-duration offset.

    The fitted model is for a per-kyr event rate,

        log(lambda_i) = beta0 + x_i beta,

    while the observed response is an event count in a finite bin,

        y_i ~ Poisson(mu_i),   mu_i = dt_i * lambda_i.
    """

    eta = beta[0] + x @ beta[1:]
    eta = np.clip(eta, -50.0, 20.0)
    log_mu = np.log(dt) + eta
    mu = np.exp(log_mu)
    return float(np.sum(y * log_mu - mu - gammaln(y + 1.0)))


def negative_binned_poisson_loglik(
    beta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dt: np.ndarray,
) -> float:
    """Numerically safe objective for scipy.optimize.minimize."""

    value = -binned_poisson_loglik(beta, x, y, dt)
    if not np.isfinite(value):
        return 1e100
    return float(value)


def fitted_rate_and_mu(beta: np.ndarray, x: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert fitted coefficients into rates per kyr and expected bin counts."""

    eta = np.clip(beta[0] + x @ beta[1:], -50.0, 20.0)
    rate = np.exp(eta)
    return rate, rate * dt

