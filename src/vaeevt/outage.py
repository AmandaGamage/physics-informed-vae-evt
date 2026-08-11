from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import genpareto


def compute_permap_outage_mask(
    snr_db: np.ndarray,
    valid_mask: np.ndarray,
    target_frac: float,
    rng: np.random.Generator | None = None,
    strict: bool = False,
) -> Tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng(0)

    valid_snr = snr_db[valid_mask]
    n_valid = valid_snr.size
    if n_valid == 0:
        return np.zeros_like(snr_db, dtype=bool), float("nan")

    n_target = max(1, int(np.ceil(n_valid * target_frac)))
    threshold = float(np.percentile(valid_snr, target_frac * 100.0))
    outage_flat = valid_snr <= threshold

    if not strict:
        n_outage = int(outage_flat.sum())
        if n_outage < n_target:
            # Rank fallback: take the n_target lowest SNR pixels.
            order = np.argsort(valid_snr)
            outage_flat = np.zeros(n_valid, dtype=bool)
            outage_flat[order[:n_target]] = True
            threshold = float(valid_snr[order[n_target - 1]])
        elif n_outage > 2 * n_target:
            # Tie block at the quantisation floor: subsample to n_target.
            idx = np.where(outage_flat)[0]
            rng.shuffle(idx)
            outage_flat = np.zeros(n_valid, dtype=bool)
            outage_flat[idx[:n_target]] = True

    outage_mask = np.zeros_like(snr_db, dtype=bool)
    outage_mask[valid_mask] = outage_flat
    return outage_mask, threshold


def fit_gpd_anchors(
    exceedances: np.ndarray, xi_clip: float = 0.5
) -> Tuple[float, float, float]:
    """Maximum-likelihood GPD fit to the pooled exceedances gamma_th - gamma.

    Returns (xi_hat, beta_hat, exceedance_p99). The location is fixed at zero,
    as required by the Pickands-Balkema-de Haan limit. These anchors are treated
    as constants during network training (paper, Section III-A5).
    """
    exceedances = np.asarray(exceedances, dtype=np.float64)
    exceedances = exceedances[exceedances > 0]

    if exceedances.size <= 100:
        return 0.1, 5.0, 10.0

    try:
        xi_hat, _loc, beta_hat = genpareto.fit(exceedances, floc=0)
        xi_hat = float(np.clip(xi_hat, -xi_clip, xi_clip))
        beta_hat = float(beta_hat)
    except Exception:  # pragma: no cover - scipy convergence failure
        xi_hat, beta_hat = 0.1, float(np.std(exceedances))

    return xi_hat, beta_hat, float(np.percentile(exceedances, 99))
