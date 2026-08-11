from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_curve,
)


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def select_tau(labels: np.ndarray, pi: np.ndarray, n_grid: int = 500) -> Tuple[float, float]:
    """Grid search for the F1-optimal routing threshold t*."""
    best_f1, best_tau = 0.0, 0.1
    for tau in np.linspace(0.001, 0.99, n_grid):
        f1 = f1_score(labels, (pi > tau).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = float(f1), float(tau)
    return best_tau, best_f1


def sharpen(pi: np.ndarray, tau: float, alpha: float = 20.0) -> np.ndarray:
    """Steep sigmoid that turns soft routing into near-binary routing."""
    return 1.0 / (1.0 + np.exp(-alpha * (pi - tau)))


def blend(mu_db: np.ndarray, tail_db: np.ndarray, pi: np.ndarray) -> np.ndarray:
    return (1.0 - pi) * mu_db + pi * tail_db


def flatten_valid(X, y, outage_mask, mu, y_tail, pi, scaler) -> Dict[str, np.ndarray]:
    """Flatten predictions to 1-D arrays over free-space pixels only, in dB."""
    building = X[:, :, :, 0].ravel()
    valid = building < 0.5
    return {
        "valid": valid,
        "y_db": scaler.inverse_transform(y.ravel()[valid]),
        "mu_db": scaler.inverse_transform(mu.ravel()[valid]),
        "tail_db": scaler.inverse_transform(y_tail.ravel()[valid]),
        "pi": pi.ravel()[valid],
        "is_los": X[:, :, :, 2].ravel()[valid] > 0.5,
        "is_outage": (outage_mask.ravel()[valid] > 0.5).astype(int),
    }


def evaluate_predictions(
    flat: Dict[str, np.ndarray],
    alpha_sharp: float = 20.0,
    n_grid: int = 500,
    tau: float | None = None,
) -> Dict:

    y = flat["y_db"]
    mu = flat["mu_db"]
    tail = flat["tail_db"]
    pi = flat["pi"]
    is_out = flat["is_outage"]
    is_los = flat["is_los"]

    if tau is None:
        tau, _ = select_tau(is_out, pi, n_grid)

    pi_s = sharpen(pi, tau, alpha_sharp)
    pred_raw = blend(mu, tail, pi)
    pred_sharp = blend(mu, tail, pi_s)

    outage_m = is_out == 1
    good_m = ~outage_m
    all_m = np.ones(len(y), dtype=bool)

    pred_cls = (pi > tau).astype(int)
    fpr, tpr, _ = roc_curve(is_out, pi)
    cm = confusion_matrix(is_out, pred_cls, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "tau": tau,
        "alpha_sharp": alpha_sharp,
        "n_outage_pixels": int(outage_m.sum()),
        "outage_fraction": float(outage_m.mean()),
        # --- RMSE (dB) ---
        "rmse_overall_mu": safe_rmse(y, mu, all_m),
        "rmse_overall_sharp": safe_rmse(y, pred_sharp, all_m),
        "rmse_outage_mu": safe_rmse(y, mu, outage_m),
        "rmse_outage_tail": safe_rmse(y, tail, outage_m),
        "rmse_outage_raw": safe_rmse(y, pred_raw, outage_m),
        "rmse_outage_sharp": safe_rmse(y, pred_sharp, outage_m),
        "rmse_good_sharp": safe_rmse(y, pred_sharp, good_m),
        "rmse_los_sharp": safe_rmse(y, pred_sharp, is_los),
        "rmse_nlos_sharp": safe_rmse(y, pred_sharp, ~is_los),
        "rmse_los_outage_sharp": safe_rmse(y, pred_sharp, is_los & outage_m),
        "rmse_nlos_outage_sharp": safe_rmse(y, pred_sharp, ~is_los & outage_m),
        "mae_overall_mu": float(mean_absolute_error(y, mu)),
        "nmse_mu": float(mean_squared_error(y, mu) / np.var(y)),
        "correlation_mu": float(np.corrcoef(y, mu)[0, 1]),
        # --- Outage classification ---
        "precision": float(precision_score(is_out, pred_cls, zero_division=0)),
        "recall": float(recall_score(is_out, pred_cls, zero_division=0)),
        "f1": float(f1_score(is_out, pred_cls, zero_division=0)),
        "roc_auc": float(auc(fpr, tpr)),
        "average_precision": float(average_precision_score(is_out, pi)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # --- Routing statistics ---
        "pi_mean_outage": float(pi[outage_m].mean()) if outage_m.any() else float("nan"),
        "pi_mean_good": float(pi[good_m].mean()) if good_m.any() else float("nan"),
        # --- Curves, for plotting ---
        "_fpr": fpr,
        "_tpr": tpr,
        "_pred_sharp": pred_sharp,
    }


def per_map_rmse(
    X, y, outage_mask, mu, y_tail, pi, scaler, tau: float, alpha_sharp: float = 20.0
) -> Dict[str, np.ndarray]:
    """Per-environment RMSE, used for the distribution plots in Fig. 2."""
    n = len(X)
    out = {
        key: np.full(n, np.nan)
        for key in (
            "rmse_overall_mu",
            "rmse_overall_sharp",
            "rmse_outage_mu",
            "rmse_outage_tail",
            "rmse_outage_raw",
            "rmse_outage_sharp",
            "rmse_good_sharp",
        )
    }
    out["n_outage_px"] = np.zeros(n, dtype=np.int32)
    out["n_valid_px"] = np.zeros(n, dtype=np.int32)

    for i in range(n):
        valid = X[i, :, :, 0] < 0.5
        if valid.sum() == 0:
            continue

        y_i = scaler.inverse_transform(y[i, :, :, 0][valid])
        mu_i = scaler.inverse_transform(mu[i, :, :, 0][valid])
        tail_i = scaler.inverse_transform(y_tail[i, :, :, 0][valid])
        pi_i = pi[i, :, :, 0][valid]
        out_i = outage_mask[i, :, :, 0][valid] > 0.5

        pi_s = sharpen(pi_i, tau, alpha_sharp)
        raw_i = blend(mu_i, tail_i, pi_i)
        sharp_i = blend(mu_i, tail_i, pi_s)
        all_m = np.ones(len(y_i), dtype=bool)

        out["rmse_overall_mu"][i] = safe_rmse(y_i, mu_i, all_m)
        out["rmse_overall_sharp"][i] = safe_rmse(y_i, sharp_i, all_m)
        out["rmse_outage_mu"][i] = safe_rmse(y_i, mu_i, out_i)
        out["rmse_outage_tail"][i] = safe_rmse(y_i, tail_i, out_i)
        out["rmse_outage_raw"][i] = safe_rmse(y_i, raw_i, out_i)
        out["rmse_outage_sharp"][i] = safe_rmse(y_i, sharp_i, out_i)
        out["rmse_good_sharp"][i] = safe_rmse(y_i, sharp_i, ~out_i)
        out["n_outage_px"][i] = int(out_i.sum())
        out["n_valid_px"][i] = int(valid.sum())

    return out
