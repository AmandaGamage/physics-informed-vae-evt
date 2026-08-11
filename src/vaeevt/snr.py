from __future__ import annotations

import numpy as np


def convert_gain_to_snr(
    gain_pixels: np.ndarray,
    noise_floor_dbm: float,
    gain_min_dbm: float = -150.0,
    gain_span_db: float = 120.0,
) -> np.ndarray:
    """Convert raw 8-bit gain pixels to SNR in dB.

    Parameters
    ----------
    gain_pixels
        Array of grayscale values in [0, 255].
    noise_floor_dbm
        Thermal noise floor P_noise in dBm.
    """
    signal_dbm = gain_min_dbm + (gain_pixels / 255.0) * gain_span_db
    return signal_dbm - noise_floor_dbm


class SNRScaler:
    """Min-max scaler between SNR in dB and the normalised [0, 1] target space.

    The bounds are the 1st and 99th percentiles of the SNR distribution, so a
    small fraction of pixels saturate at 0 and 1 after `transform`
    """

    def __init__(self, snr_min: float, snr_max: float):
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)

    @property
    def snr_range(self) -> float:
        return self.snr_max - self.snr_min

    def transform(self, snr_db) -> np.ndarray:
        snr_db = np.asarray(snr_db, dtype=np.float32)
        return np.clip((snr_db - self.snr_min) / self.snr_range, 0.0, 1.0)

    def inverse_transform(self, snr_norm) -> np.ndarray:
        snr_norm = np.asarray(snr_norm, dtype=np.float32)
        return snr_norm * self.snr_range + self.snr_min

    @classmethod
    def from_samples(
        cls, all_snr_db: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0
    ) -> "SNRScaler":
        return cls(
            float(np.percentile(all_snr_db, lo_pct)),
            float(np.percentile(all_snr_db, hi_pct)),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"SNRScaler(min={self.snr_min:.1f} dB, max={self.snr_max:.1f} dB)"
