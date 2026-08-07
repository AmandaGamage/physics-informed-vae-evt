"""Figure generation: input channels, sample predictions, SNR distributions.

Reproduces the qualitative figures in the paper (Fig. 2, 3 and 4). Every
function takes already-computed arrays so that figures can be regenerated from
the exported CSVs without re-running the model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .features import CHANNEL_NAMES
from .metrics import blend, sharpen

COLOR_EMPIRICAL = "#D46A1A"
COLOR_PREDICTED = "#4BAF82"
COLOR_TAIL_EMPIRICAL = "#7A1500"
COLOR_TAIL_PREDICTED = "#1A5E3A"


def plot_input_channels(X, index: int, save_path=None):
    """Show all ten channels of X_geo for one environment."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ch, ax in enumerate(axes.ravel()):
        cmap = "gray" if ch == 0 else "viridis"
        im = ax.imshow(X[index, :, :, ch], cmap=cmap, vmin=0, vmax=1, origin="lower")
        ax.set_title(f"ch{ch}: {CHANNEL_NAMES[ch]}", fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Physics-informed input tensor X_geo", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_predictions(
    X, y, outage_mask, mu, y_tail, pi, scaler, tau, alpha=20.0, n=4, save_path=None
):
    """Ground truth, bulk mean, routed prediction, error, routing mask."""
    n = min(n, len(X))
    fig, axes = plt.subplots(n, 6, figsize=(28, 4.5 * n))
    axes = np.atleast_2d(axes)

    for i in range(n):
        building = X[i, :, :, 0] > 0.5
        valid = ~building
        los = X[i, :, :, 2]

        y_db = scaler.inverse_transform(y[i, :, :, 0])
        mu_db = scaler.inverse_transform(mu[i, :, :, 0])
        tail_db = scaler.inverse_transform(y_tail[i, :, :, 0])
        pi_i = pi[i, :, :, 0]
        pi_s = sharpen(pi_i, tau, alpha)
        pred = blend(mu_db, tail_db, pi_s)
        vmin, vmax = np.percentile(y_db[valid], [1, 99])

        los_rgb = np.zeros((*los.shape, 3))
        los_rgb[los > 0.5] = [0.2, 0.8, 0.2]
        los_rgb[(los < 0.5) & valid] = [0.8, 0.2, 0.2]
        los_rgb[building] = [0.3, 0.3, 0.3]

        panels = [
            (los_rgb, "LoS (green) / NLoS (red)", None, None, None),
            (y_db, "Ground truth SNR", "viridis", vmin, vmax),
            (mu_db, "Bulk mean mu", "viridis", vmin, vmax),
            (pred, "Routed prediction", "viridis", vmin, vmax),
            (pred - y_db, "Error", "RdBu_r", None, None),
            (outage_mask[i, :, :, 0], "True outage region", "Reds", 0, 1),
        ]
        for j, (data, title, cmap, vlo, vhi) in enumerate(panels):
            ax = axes[i, j]
            if cmap is None:
                ax.imshow(data, origin="lower")
            elif title == "Error":
                lim = max(abs(np.percentile(data[valid], [5, 95])))
                ax.imshow(
                    np.ma.masked_where(building, data),
                    cmap=cmap, origin="lower", vmin=-lim, vmax=lim,
                )
            else:
                ax.imshow(
                    np.ma.masked_where(building, data),
                    cmap=cmap, origin="lower", vmin=vlo, vmax=vhi,
                )
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_snr_distribution(
    snr_true_db, snr_pred_db, threshold_db, n_bins=55, title="", save_path=None
):
    """Empirical vs predicted SNR density, with a log-scale zoom on the tail.

    This is the figure that shows whether the model reproduces the bimodal
    bulk (LoS peak plus NLoS peak) and the shape of the lower tail.
    """
    lo = min(snr_true_db.min(), snr_pred_db.min()) - 2
    hi = max(snr_true_db.max(), snr_pred_db.max()) + 2
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    dens_true, _ = np.histogram(snr_true_db, bins=edges, density=True)
    dens_pred, _ = np.histogram(snr_pred_db, bins=edges, density=True)
    tail = centers <= threshold_db

    fig, (ax, ax_zoom) = plt.subplots(
        2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.45}
    )

    for target in (ax, ax_zoom):
        target.bar(centers, dens_true, width=width, color=COLOR_EMPIRICAL,
                   alpha=0.85, label="Empirical", zorder=2)
        target.bar(centers, dens_pred, width=width, color=COLOR_PREDICTED,
                   alpha=0.75, label="Predicted", zorder=3)
        if tail.any():
            target.bar(centers[tail], dens_true[tail], width=width,
                       color=COLOR_TAIL_EMPIRICAL, zorder=4)
            target.bar(centers[tail], dens_pred[tail], width=width,
                       color=COLOR_TAIL_PREDICTED, alpha=0.9, zorder=5)
        target.axvline(threshold_db, color="#222222", linestyle="--", lw=1.2, zorder=6)
        target.grid(True, alpha=0.18, linestyle=":", zorder=0)

    ax.set_xlim(lo, hi)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Probability density")
    ax.set_title(title or f"SNR distribution (threshold = {threshold_db:.1f} dB)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    if tail.any():
        zoom_lo = centers[tail].min() - 2 * width
        zoom_hi = threshold_db + (threshold_db - zoom_lo) * 0.18 + 2 * width
        ax_zoom.set_xlim(zoom_lo, zoom_hi)
        in_range = (centers >= zoom_lo) & (centers <= zoom_hi)
        nonzero = np.concatenate(
            [dens_true[in_range & (dens_true > 0)], dens_pred[in_range & (dens_pred > 0)]]
        )
        if nonzero.size:
            ax_zoom.set_ylim(max(nonzero.min() * 0.05, 1e-7), nonzero.max() * 6)
    ax_zoom.set_yscale("log")
    ax_zoom.set_xlabel("SNR (dB)")
    ax_zoom.set_ylabel("Density (log)")
    ax_zoom.set_title("Tail zoom", fontsize=10, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_per_map_rmse(per_map_by_threshold: dict, save_path=None):
    """Distribution of outage RMSE across test environments (Fig. 2)."""
    labels = list(per_map_by_threshold)
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5))
    axes = np.atleast_1d(axes)

    for ax, label in zip(axes, labels):
        values = per_map_by_threshold[label]
        values = values[~np.isnan(values)]
        ax.hist(values, bins=40, color="#6A1B9A", alpha=0.75)
        ax.axvline(values.mean(), color="black", linestyle="--",
                   label=f"mean = {values.mean():.2f} dB")
        ax.set_xlabel("Outage RMSE (dB)")
        ax.set_ylabel("Number of test maps")
        ax.set_title(f"Threshold = {label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
