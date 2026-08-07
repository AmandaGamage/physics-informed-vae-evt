"""Central configuration for the Physics-informed VAE-EVT framework.

Every magic number used anywhere in the pipeline lives here so that a reader can
see the full experimental setup in one place, and so that a reviewer can change
a setting without editing model code.

Values are the ones used to produce the results reported in the paper.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    radiomapseer_dir: str = "data/RadioMapSeer"
    propagation_model: str = "DPM"  # subdirectory of gain/ to read
    n_maps: int = 300  # number of building layouts, N_map
    n_tx_per_map: int = 10  # transmitter positions per layout, N_tx
    map_size: int = 256  # M in the paper (M x M grid)

    # ------------------------------------------------------------------
    # Link budget -> SNR conversion  (paper, Section II-A)
    # ------------------------------------------------------------------
    bandwidth_hz: float = 10e6  # B
    n0_dbm_hz: float = -174.0  # N0
    noise_figure_db: float = 0.0  # NF
    # RadioMapSeer stores path gain as 8-bit grayscale. We map pixel value
    # g in [0, 255] linearly onto received power in dBm:
    #     P_rx = gain_min_dbm + (g / 255) * gain_span_db
    gain_min_dbm: float = -150.0
    gain_span_db: float = 120.0

    # SNR normalisation percentiles used to build the [0, 1] target scale.
    snr_lo_percentile: float = 1.0
    snr_hi_percentile: float = 99.0

    # ------------------------------------------------------------------
    # Outage definition  (paper, Section II-A and III-A5)
    # ------------------------------------------------------------------
    # q in the paper, expressed as a fraction. 0.001 = 0.1% SNR quantile.
    target_outage_frac: float = 0.001

    # ------------------------------------------------------------------
    # Physics-informed preprocessing  (paper, Section III-A)
    # ------------------------------------------------------------------
    los_ray_steps: int = 100  # samples per LoS ray
    shadow_steps: int = 30  # samples for the localised shadow score, N_s
    nlos_max_depth: float = 10.0  # d_max in Eq. (6)
    tx_proximity_sigma_sq: float = 8.0  # 2 * sigma_T^2 in T_x(p)
    tx_proximity_cutoff_sq: float = 30.25  # truncation radius^2 (5.5 px)
    edge_weight_los: float = 0.6  # alpha_1 in E(p)
    edge_weight_shadow: float = 0.4  # alpha_2 in E(p)
    edge_blur_sigma: float = 1.0  # sigma_E in G_{sigma_E}
    edge_gain: float = 1.5  # contrast gain applied before clipping
    prior_a1: float = 0.3  # a_1 in Eq. (7), LoS branch
    prior_b1: float = 0.5  # b_1 in Eq. (7), shadow term
    prior_b2: float = 0.3  # b_2 in Eq. (7), penetration-depth term
    prior_b3: float = 0.2  # b_3 in Eq. (7), distance term
    prior_clip_lo: float = 0.01  # epsilon in Eq. (7)
    prior_clip_hi: float = 0.95

    # ------------------------------------------------------------------
    # Model  (paper, Section III-B/C/D)
    # ------------------------------------------------------------------
    latent_dim: int = 16  # d_z = d_t
    encoder_widths: Tuple[int, ...] = (32, 64, 128, 256)
    bottleneck: int = 16  # 256 / 2^4
    gpd_xi_clip: float = 0.4  # numerical clamp on xi inside the network
    gpd_fit_xi_clip: float = 0.5  # clamp applied to the MLE fit
    pi_temperature_init: float = 0.8
    pi_temperature_range: Tuple[float, float] = (0.3, 1.5)
    pi_prior_rate: float = 0.02  # bias init for the outage logit

    # ------------------------------------------------------------------
    # Training  (paper, Section IV-A2)
    # ------------------------------------------------------------------
    batch_size: int = 4
    epochs: int = 50
    learning_rate: float = 1e-4
    seed: int = 42
    test_size: float = 0.2
    grad_clip_norm: float = 1.0

    # Initial loss weights (the schedules below override during training).
    lambda_recon: float = 1.0
    lambda_pi: float = 0.5
    lambda_outage_recon: float = 1.0
    lambda_pi_sharp: float = 0.1
    lambda_los_outage: float = 1.0
    kl_weight_bulk: float = 0.0
    kl_weight_tail: float = 0.0

    # Weight schedules: (warmup_epoch, ramp_end_epoch, start, end)
    kl_schedule: Tuple[int, float, float] = (15, 0.3, 0.2)  # warmup, max_b, max_t
    pi_schedule: Tuple[int, int, float, float] = (5, 20, 0.5, 5.0)
    outage_recon_schedule: Tuple[int, int, float, float] = (3, 20, 1.0, 10.0)
    pi_sharp_schedule: Tuple[int, int, float, float] = (10, 35, 0.1, 3.0)

    # Fixed multipliers inside the composite loss (see losses.py).
    tail_mse_scale: float = 10.0
    focal_gamma: float = 3.0
    focal_alpha_outage: float = 2.0
    focal_alpha_good: float = 1.5
    dice_weight: float = 3.0
    calibration_weight: float = 5.0
    separation_margin: float = 0.4
    separation_weight: float = 10.0
    sharp_weight_outage: float = 3.0

    # ------------------------------------------------------------------
    # Evaluation  (paper, Section III-D3 and IV)
    # ------------------------------------------------------------------
    alpha_sharp: float = 20.0  # kappa, steepness of the sharpening sigmoid
    tau_grid_size: int = 500  # grid used to select t*
    eval_thresholds: Tuple[float, ...] = (0.001, 0.01, 0.10)
    inference_batch_size: int = 8

    # --- Routing threshold t* -----------------------------------------
    # `tau_selection` controls how the routing threshold is obtained:
    #
    #   "global"         select t* ONCE, by maximising F1 at the training
    #                    outage quantile, then reuse that single value at every
    #                    evaluation threshold and in every figure. This is the
    #                    convention used by the CSV export that produced the
    #                    published tables and plots.
    #
    #   "per_threshold"  re-select t* independently at each evaluation
    #                    threshold. Retained for comparison only. It makes the
    #                    reported outage RMSE depend strongly on the threshold
    #                    (t* ranged from 0.0030 to 0.9900 across the published
    #                    runs), so numbers are not comparable across rows.
    #
    # Set `tau` to a float to pin an explicit value and skip selection
    # entirely -- e.g. tau = 0.4192 reproduces the exported 1%-trained CSVs.
    tau_selection: str = "global"
    tau: float | None = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    output_dir: str = "runs/default"

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def noise_floor_dbm(self) -> float:
        """P_noise = 10*log10(B) + N0 + NF  (paper, Section II-A)."""
        return (
            10.0 * np.log10(self.bandwidth_hz) + self.n0_dbm_hz + self.noise_figure_db
        )

    @property
    def buildings_dir(self) -> Path:
        return Path(self.radiomapseer_dir) / "png" / "buildings_complete"

    @property
    def antennas_dir(self) -> Path:
        return Path(self.radiomapseer_dir) / "png" / "antennas"

    @property
    def gain_dir(self) -> Path:
        return Path(self.radiomapseer_dir) / "gain" / self.propagation_model

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, default=list)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path) as fh:
            payload = json.load(fh)
        for key in ("encoder_widths", "pi_temperature_range", "eval_thresholds",
                    "kl_schedule", "pi_schedule", "outage_recon_schedule",
                    "pi_sharp_schedule"):
            if key in payload and isinstance(payload[key], list):
                payload[key] = tuple(payload[key])
        return cls(**payload)


def set_global_seed(seed: int) -> None:
    """Seed numpy and TensorFlow. Import TF lazily so that feature-only
    workflows do not pay the import cost."""
    import random

    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
