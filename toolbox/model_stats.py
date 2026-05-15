"""Small statistical helpers shared by the analysis scripts.

The project fits custom Poisson likelihoods because the response is a sparse
event-count series with a bin-duration offset and, in some scripts, event
history terms recomputed from the catalogue. Common Python packages provide the
optimizer and chi-square distribution, but AICc and likelihood-gain summaries
still need to be computed from the fitted log likelihoods. Keeping those
formulae here avoids repeating them across scripts.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def information_criteria(log_likelihood: float, n_params: int, n_obs: int) -> dict[str, float]:
    """Return AIC, AICc, and BIC for a fitted likelihood model."""

    k = int(n_params)
    n = int(n_obs)
    aic = 2.0 * k - 2.0 * float(log_likelihood)
    aicc = aic + (2.0 * k * (k + 1.0)) / max(n - k - 1.0, 1.0)
    bic = np.log(n) * k - 2.0 * float(log_likelihood)
    return {"AIC": float(aic), "AICc": float(aicc), "BIC": float(bic)}


def likelihood_gain(loglik_full: float, loglik_reduced: float) -> float:
    """Log-likelihood gain in nats for a nested model comparison."""

    return float(loglik_full - loglik_reduced)


def likelihood_ratio_p_value(ll_gain: float, df: int) -> tuple[float, float]:
    """Return the likelihood-ratio statistic and chi-square p value."""

    lr_stat = 2.0 * float(ll_gain)
    p_value = float(chi2.sf(max(lr_stat, 0.0), int(df)))
    return float(lr_stat), p_value


def bits_from_loglik_gain(ll_gain: float, denominator: int | float) -> float:
    """Convert a log-likelihood gain in nats to bits per denominator unit."""

    return float(ll_gain / np.log(2.0) / max(float(denominator), 1.0))


def nested_likelihood_metrics(
    *,
    loglik_full: float,
    loglik_reduced: float,
    df: int,
    n_bins: int,
    n_events: int,
    aicc_full: float,
    aicc_reduced: float,
) -> dict[str, float | int]:
    """Summary metrics used repeatedly for nested predictive-model tests."""

    ll_gain = likelihood_gain(loglik_full, loglik_reduced)
    lr_stat, p_value = likelihood_ratio_p_value(ll_gain, df)
    return {
        "df": int(df),
        "n_bins": int(n_bins),
        "n_events": int(n_events),
        "loglik_reduced": float(loglik_reduced),
        "loglik_full": float(loglik_full),
        "ll_gain_nats": ll_gain,
        "info_bits_per_event": bits_from_loglik_gain(ll_gain, n_events),
        "info_bits_per_bin": bits_from_loglik_gain(ll_gain, n_bins),
        "LR_statistic": lr_stat,
        "LR_p_value": p_value,
        "delta_AICc_full_minus_reduced": float(aicc_full - aicc_reduced),
    }

