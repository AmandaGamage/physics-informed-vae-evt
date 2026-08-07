"""Per-map outage labelling and the dataset-level GPD anchors.

Section II-A of the paper defines the outage region as

    O = {p in Omega : gamma(p) < gamma_th},

with gamma_th the q-th percentile of the per-map SNR distribution over the
free-space pixels Omega.

`compute_permap_outage_mask` implements this, plus two guards that the
published experiments relied on. Both are consequences of the 8-bit
quantisation of the RadioMapSeer gain maps, which produces large ties at the
low end of the SNR distribution:

  1. If strictly fewer than ceil(|Omega| * q) pixels fall below the percentile,
     the n lowest-ranked pixels are taken instead (rank fallback).
  2. If more than twice that many pixels fall at or below the percentile
     (i.e. a large tie block at the quantisation floor), a random subset of
     size n is retained.

Guard (2) means the realised label set is a random subset of {gamma <= gamma_th}
rather than the whole set. This is a deviation from Eq. (2) and is documented in
KNOWN_DEVIATIONS.md. It is reproduced here because it is what produced the
published numbers; pass `strict=True` to disable both guards.
"""

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
    """Return (boolean outage mask over the full grid, threshold in dB).

    Parameters
    ----------
    snr_db
        (M, M) SNR map in dB.
    valid_mask
        (M, M) boolean, True on free-space pixels (Omega).
    target_frac
        q expressed as a fraction, e.g. 0.001 for the 0.1% quantile.
    rng
        Generator used by the tie-breaking subsample. Seeded for reproducibility.
    strict
        If True, return exactly {gamma <= gamma_th} with no rank fallback and
        no tie subsampling. This follows Eq. (2) literally but does not
        reproduce the published label counts.
    """
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
