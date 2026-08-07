"""RadioMapSeer loading and dataset assembly.

Expected layout on disk (as distributed with the dataset):

    <radiomapseer_dir>/
        png/buildings_complete/<map>.png
        png/antennas/<map>_<tx>.png
        gain/DPM/<map>_<tx>.png

Produces:
    X            (N, 256, 256, 10) float32   physics-informed input tensor
    y            (N, 256, 256,  1) float32   normalised ground-truth SNR
    outage_mask  (N, 256, 256,  1) float32   binary outage labels
    scaler       SNRScaler
    evt_params   dict with the GPD anchors and threshold statistics
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .features import (
    compute_geometric_features,
    find_tx_from_antenna_file,
    find_tx_from_signal,
    stack_input_tensor,
)
from .outage import compute_permap_outage_mask, fit_gpd_anchors
from .snr import SNRScaler, convert_gain_to_snr


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    outage_mask: np.ndarray
    scaler: SNRScaler
    evt_params: Dict
    thresholds_db: np.ndarray

    def split(self, test_size: float, seed: int):
        idx = np.arange(len(self.X))
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed
        )
        return train_idx, test_idx


# ----------------------------------------------------------------------
# Raw loading
# ----------------------------------------------------------------------
def _load_one_sample(args):
    map_idx, tx_idx, cfg, building_map = args
    gain_path = cfg.gain_dir / f"{map_idx}_{tx_idx}.png"
    antenna_path = cfg.antennas_dir / f"{map_idx}_{tx_idx}.png"
    if not gain_path.exists():
        return None

    with Image.open(gain_path) as img:
        gain_pixels = np.array(img.convert("L"), dtype=np.float32)

    snr_db = convert_gain_to_snr(
        gain_pixels, cfg.noise_floor_dbm, cfg.gain_min_dbm, cfg.gain_span_db
    )

    tx_y, tx_x = find_tx_from_antenna_file(antenna_path)
    if tx_y is None:
        tx_y, tx_x = find_tx_from_signal(gain_pixels, building_map)

    features = compute_geometric_features(tx_y, tx_x, building_map, cfg)
    return {"snr_db": snr_db, "tx_pos": (tx_y, tx_x), "features": features}


def load_radiomapseer(cfg, verbose: bool = True) -> Tuple[List[Dict], np.ndarray]:
    """Load N_map layouts x N_tx transmitters and compute their geometry."""
    building_files = sorted(cfg.buildings_dir.glob("*.png"))
    map_indices = sorted(int(p.stem) for p in building_files)
    if not map_indices:
        raise FileNotFoundError(
            f"No building maps found in {cfg.buildings_dir}. "
            "Check Config.radiomapseer_dir."
        )

    samples: List[Dict] = []
    all_snr: List[np.ndarray] = []
    t0 = time.time()

    for i, map_idx in enumerate(map_indices[: cfg.n_maps]):
        building_path = cfg.buildings_dir / f"{map_idx}.png"
        if not building_path.exists():
            continue
        with Image.open(building_path) as img:
            building_map = (
                np.array(img.convert("L"), dtype=np.float32) / 255.0 > 0.5
            ).astype(np.float32)

        work = [
            (map_idx, tx_idx, cfg, building_map)
            for tx_idx in range(cfg.n_tx_per_map)
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(_load_one_sample, work):
                if result is None:
                    continue
                valid = result["features"]["building_map"] < 0.5
                all_snr.append(result["snr_db"][valid].ravel())
                samples.append(result)

        if verbose and (i + 1) % 20 == 0:
            gc.collect()
            print(
                f"  {i + 1}/{cfg.n_maps} layouts, {len(samples)} samples "
                f"({time.time() - t0:.1f}s)"
            )

    if verbose:
        print(f"\n  {len(samples)} samples loaded in {time.time() - t0:.1f}s")

    gc.collect()
    return samples, np.concatenate(all_snr)


# ----------------------------------------------------------------------
# Tensor assembly
# ----------------------------------------------------------------------
def build_dataset(samples: List[Dict], all_snr: np.ndarray, cfg, verbose: bool = True):
    """Assemble X, y and outage labels, and fit the dataset-level GPD anchors."""
    scaler = SNRScaler.from_samples(
        all_snr, cfg.snr_lo_percentile, cfg.snr_hi_percentile
    )
    if verbose:
        print(f"  SNR bounds: [{scaler.snr_min:.1f}, {scaler.snr_max:.1f}] dB")

    n = len(samples)
    M = cfg.map_size
    X = np.zeros((n, M, M, 10), dtype=np.float32)
    y = np.zeros((n, M, M, 1), dtype=np.float32)
    outage = np.zeros((n, M, M, 1), dtype=np.float32)

    rng = np.random.default_rng(cfg.seed)
    thresholds: List[float] = []
    exceedances: List[float] = []
    n_los = n_nlos = n_valid_total = 0
    los_outage = nlos_outage = 0

    for i, sample in enumerate(samples):
        feats = sample["features"]
        snr_db = sample["snr_db"]
        building_map = feats["building_map"]
        valid = building_map < 0.5

        mask, threshold = compute_permap_outage_mask(
            snr_db, valid, cfg.target_outage_frac, rng=rng
        )
        thresholds.append(threshold)
        outage[i, :, :, 0] = mask.astype(np.float32)

        exc = threshold - snr_db[mask]
        exceedances.extend(exc[exc > 0].tolist())

        X[i] = stack_input_tensor(feats, float(scaler.transform(threshold)))
        y[i, :, :, 0] = scaler.transform(snr_db)

        los = feats["los_mask"] > 0.5
        n_los += int(los.sum())
        n_nlos += int(((~los) & valid).sum())
        n_valid_total += int(valid.sum())
        los_outage += int((mask & los).sum())
        nlos_outage += int((mask & (~los) & valid).sum())

        if verbose and (i + 1) % 500 == 0:
            print(f"    {i + 1}/{n} samples processed")

    thresholds = np.asarray(thresholds, dtype=np.float32)
    total_outage = float(outage.sum())
    empirical_frac = total_outage / max(n_valid_total, 1)

    xi_hat, beta_hat, exc_p99 = fit_gpd_anchors(
        np.asarray(exceedances), cfg.gpd_fit_xi_clip
    )

    if verbose:
        print(
            f"\n  LoS: {n_los:,} ({100 * n_los / n_valid_total:.1f}%), "
            f"NLoS: {n_nlos:,} ({100 * n_nlos / n_valid_total:.1f}%)"
        )
        print(
            f"  Outage in LoS: {los_outage:,} "
            f"({100 * los_outage / max(total_outage, 1):.1f}%), "
            f"in NLoS: {nlos_outage:,} "
            f"({100 * nlos_outage / max(total_outage, 1):.1f}%)"
        )
        print(
            f"  Threshold range: [{thresholds.min():.1f}, {thresholds.max():.1f}] dB"
        )
        print(f"  Empirical outage fraction: {100 * empirical_frac:.2f}%")
        print(f"  GPD anchors: xi={xi_hat:.3f}, beta={beta_hat:.3f} dB")

    evt_params = {
        "empirical_xi": xi_hat,
        # beta and the exceedance cap are expressed in the normalised target
        # space, because the tail head operates on normalised SNR.
        "empirical_beta": beta_hat / scaler.snr_range,
        "empirical_exc_max": exc_p99 / scaler.snr_range,
        "empirical_beta_db": beta_hat,
        "empirical_outage_frac": empirical_frac,
        "target_outage_frac": cfg.target_outage_frac,
        "avg_threshold_db": float(thresholds.mean()),
        "avg_threshold_norm": float(scaler.transform(thresholds.mean())),
        "snr_min": scaler.snr_min,
        "snr_max": scaler.snr_max,
        "los_frac": n_los / max(n_valid_total, 1),
    }

    samples.clear()
    gc.collect()

    return Dataset(X, y, outage, scaler, evt_params, thresholds)


def rebuild_for_threshold(
    X: np.ndarray,
    y: np.ndarray,
    scaler: SNRScaler,
    target_frac: float,
    seed: int = 0,
):
    """Re-label the outage region at a different quantile and rewrite channel 9.

    This is how the paper evaluates a single trained model at several outage
    thresholds without retraining: the threshold is an explicit conditioning
    input, so only channel 9 and the labels change.
    """
    rng = np.random.default_rng(seed)
    X_new = X.copy()
    n = len(X)
    outage = np.zeros((n, X.shape[1], X.shape[2], 1), dtype=np.float32)
    thresholds = np.zeros(n, dtype=np.float32)

    for i in range(n):
        valid = X[i, :, :, 0] < 0.5
        snr_db = scaler.inverse_transform(y[i, :, :, 0])
        mask, threshold = compute_permap_outage_mask(
            snr_db, valid, target_frac, rng=rng
        )
        outage[i, :, :, 0] = mask.astype(np.float32)
        thresholds[i] = threshold
        X_new[i, :, :, 9] = float(scaler.transform(threshold))

    return X_new, outage, thresholds
