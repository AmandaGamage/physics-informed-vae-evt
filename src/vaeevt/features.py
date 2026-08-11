from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

CHANNEL_NAMES = (
    "building_map",
    "tx_map",
    "los_mask",
    "los_dist",
    "nlos_shadow",
    "nlos_depth",
    "shadow_edge",
    "outage_prior",
    "dist_all",
    "threshold",
)
N_CHANNELS = len(CHANNEL_NAMES)


# ----------------------------------------------------------------------
# Ray tracing
# ----------------------------------------------------------------------
def compute_los_mask(
    tx_y: int,
    tx_x: int,
    building_map: np.ndarray,
    map_size: int = 256,
    n_steps: int = 100,
):
    """Discretised ray trace from p_tx to every pixel.

    Returns (los_mask, n_hits) where `n_hits` counts building intersections
    along the ray. A pixel is LoS iff n_hits == 0.

    Note on sampling density: with n_steps samples the inter-sample spacing on
    the longest diagonal (256*sqrt(2) ~ 362 px) is ~362/n_steps px. n_steps=100
    therefore gives roughly 3.6 px spacing, not sub-pixel spacing. See
    KNOWN_DEVIATIONS.md.
    """
    y_coords, x_coords = np.ogrid[:map_size, :map_size]
    ray_dy = y_coords - tx_y
    ray_dx = x_coords - tx_x
    t = np.arange(1, n_steps, dtype=np.float32) / n_steps

    sample_y = np.clip(
        (tx_y + t[:, None, None] * ray_dy[None]).astype(np.int32), 0, map_size - 1
    )
    sample_x = np.clip(
        (tx_x + t[:, None, None] * ray_dx[None]).astype(np.int32), 0, map_size - 1
    )

    n_hits = (building_map[sample_y, sample_x] > 0.5).astype(np.float32).sum(axis=0)
    los_mask = (n_hits == 0).astype(np.float32)
    los_mask[building_map > 0.5] = 0.0
    return los_mask, n_hits


def compute_nlos_depth(
    n_hits: np.ndarray, building_map: np.ndarray, max_depth: float = 10.0
) -> np.ndarray:
    """Eq. (6): normalised wall-penetration depth min(n_hits / d_max, 1)."""
    depth = np.clip(n_hits / max_depth, 0.0, 1.0).astype(np.float32)
    depth[building_map > 0.5] = 0.0
    return depth


def compute_shadow_map(
    tx_y: int,
    tx_x: int,
    building_map: np.ndarray,
    map_size: int = 256,
    n_steps: int = 30,
) -> np.ndarray:
    """Localised shadowing score S(p): fraction of samples along the segment
    from p toward p_tx that fall inside a building.

    Uses a coarser discretisation than the LoS trace (n_steps=30 vs 100), which
    makes it sensitive to bulk obstruction near p rather than to thin walls far
    along the path.
    """
    y_coords, x_coords = np.ogrid[:map_size, :map_size]
    ray_dy = tx_y - y_coords
    ray_dx = tx_x - x_coords
    t = np.arange(1, n_steps, dtype=np.float32) / n_steps

    sample_y = np.clip(
        (y_coords[None] + t[:, None, None] * ray_dy[None]).astype(np.int32),
        0,
        map_size - 1,
    )
    sample_x = np.clip(
        (x_coords[None] + t[:, None, None] * ray_dx[None]).astype(np.int32),
        0,
        map_size - 1,
    )

    shadow = (building_map[sample_y, sample_x] > 0.5).astype(np.float32).sum(axis=0)
    shadow = np.clip(shadow / n_steps, 0.0, 1.0)
    shadow[building_map > 0.5] = 0.0
    return shadow.astype(np.float32)


# ----------------------------------------------------------------------
# Full descriptor stack
# ----------------------------------------------------------------------
def compute_geometric_features(
    tx_y: int,
    tx_x: int,
    building_map: np.ndarray,
    cfg,
) -> Dict[str, np.ndarray]:
    """Compute channels 0-8 of X_geo for one (B, p_tx) pair."""
    M = cfg.map_size
    yy, xx = np.ogrid[:M, :M]
    dist_sq = ((yy - tx_y).astype(np.float32) ** 2) + (
        (xx - tx_x).astype(np.float32) ** 2
    )

    # --- Transmitter proximity: truncated Gaussian centred at p_tx ---
    tx_map = np.where(
        dist_sq <= cfg.tx_proximity_cutoff_sq,
        np.exp(-dist_sq / cfg.tx_proximity_sigma_sq),
        0.0,
    ).astype(np.float32)

    # --- Distance channels ---
    dist = np.maximum(np.sqrt(dist_sq), 1.0)
    dist_norm = (dist / (np.sqrt(2.0) * M)).astype(np.float32)
    log_dist = np.log10(dist)
    ld_min, ld_max = log_dist.min(), log_dist.max()
    log_dist_norm = ((log_dist - ld_min) / (ld_max - ld_min + 1e-8)).astype(np.float32)

    # --- LoS / NLoS segregation ---
    los_mask, n_hits = compute_los_mask(
        tx_y, tx_x, building_map, M, cfg.los_ray_steps
    )
    nlos_mask = ((los_mask < 0.5) & (building_map < 0.5)).astype(np.float32)

    # Eq. (5): LoS-masked log distance
    los_dist = log_dist_norm * los_mask

    # --- Shadowing and penetration depth (NLoS only) ---
    shadow_full = compute_shadow_map(tx_y, tx_x, building_map, M, cfg.shadow_steps)
    nlos_shadow = shadow_full * nlos_mask
    nlos_depth = compute_nlos_depth(n_hits, building_map, cfg.nlos_max_depth) * nlos_mask

    # --- Edge map E(p) guiding the decoder's spatial attention ---
    los_edge = np.sqrt(sobel(los_mask, axis=0) ** 2 + sobel(los_mask, axis=1) ** 2)
    blurred = gaussian_filter(shadow_full, sigma=cfg.edge_blur_sigma)
    shadow_grad = np.sqrt(sobel(blurred, axis=0) ** 2 + sobel(blurred, axis=1) ** 2)
    edge = cfg.edge_weight_los * los_edge + cfg.edge_weight_shadow * shadow_grad
    emax = edge.max()
    if emax > 1e-8:
        edge = edge / emax
    edge = np.clip(edge * cfg.edge_gain, 0.0, 1.0).astype(np.float32)
    edge[building_map > 0.5] = 0.0

    # --- Eq. (7): coarse geometric outage prior ---
    dist_prior = (log_dist - ld_min) / (ld_max - ld_min + 1e-8)
    prior = cfg.prior_a1 * dist_prior * los_mask + (
        cfg.prior_b1 * nlos_shadow
        + cfg.prior_b2 * nlos_depth
        + cfg.prior_b3 * dist_prior
    ) * nlos_mask
    prior[building_map > 0.5] = 0.0
    prior = np.clip(prior, cfg.prior_clip_lo, cfg.prior_clip_hi).astype(np.float32)

    # --- Distance on all valid pixels (the tail branch needs this for LoS
    #     outages, which the LoS-masked channel alone cannot express) ---
    dist_all = dist_norm.copy()
    dist_all[building_map > 0.5] = 0.0

    return {
        "building_map": building_map.astype(np.float32),
        "tx_map": tx_map,
        "los_mask": los_mask,
        "los_dist": los_dist.astype(np.float32),
        "nlos_shadow": nlos_shadow.astype(np.float32),
        "nlos_depth": nlos_depth.astype(np.float32),
        "shadow_edge": edge,
        "outage_prior": prior,
        "dist_all": dist_all,
        # extras, useful for figures but not fed to the network
        "nlos_mask": nlos_mask,
        "shadow_full": shadow_full,
        "n_hits": n_hits,
    }


def stack_input_tensor(
    features: Dict[str, np.ndarray], threshold_norm: float
) -> np.ndarray:
    """Assemble the (M, M, 10) input tensor X_geo in the canonical order."""
    M = features["building_map"].shape[0]
    channels = [features[name] for name in CHANNEL_NAMES[:-1]]
    channels.append(np.full((M, M), threshold_norm, dtype=np.float32))
    return np.stack(channels, axis=-1).astype(np.float32)


# ----------------------------------------------------------------------
# Transmitter localisation
# ----------------------------------------------------------------------
def find_tx_from_antenna_file(antenna_path) -> tuple:
    """Centroid of the bright pixels in a RadioMapSeer antenna PNG."""
    import os

    from PIL import Image

    if not os.path.exists(antenna_path):
        return None, None
    with Image.open(antenna_path) as img:
        ant = np.array(img.convert("L"))
    ys, xs = np.where(ant > 128)
    if ys.size == 0:
        return None, None
    return int(ys.mean()), int(xs.mean())


def find_tx_from_signal(
    signal_pixels: np.ndarray, building_map: np.ndarray | None = None, margin: int = 5
) -> tuple:
    """Fallback: locate the transmitter at the peak of the gain map, ignoring
    a border margin and any pixel inside a building."""
    h, w = signal_pixels.shape
    search = np.ones((h, w), dtype=bool)
    search[:margin, :] = search[-margin:, :] = False
    search[:, :margin] = search[:, -margin:] = False
    if building_map is not None:
        search &= building_map < 0.5

    arr = signal_pixels.astype(np.float32).copy()
    arr[~search] = -999.0
    y, x = np.unravel_index(np.argmax(arr), arr.shape)
    return int(y), int(x)
