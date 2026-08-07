"""Loss-weight schedules and training monitors (paper, Section IV-A2).

The composite objective is not stable if every term is switched on at full
weight from epoch 0: the KL terms cause posterior collapse, and the
classification and sharpening terms drive pi to a degenerate constant before
the reconstruction branches have learned anything. All four schedules below
implement the "KL warmup ... with progressive ramping of the classification,
outage reconstruction, and sharpening terms" described in the paper.
"""

from __future__ import annotations

import time

import numpy as np
from tensorflow import keras


def _ramp(epoch: int, warmup: int, ramp_end: int, start: float, end: float) -> float:
    if epoch < warmup:
        return start
    if epoch < ramp_end:
        progress = (epoch - warmup) / max(ramp_end - warmup, 1)
        return start + (end - start) * progress
    return end


class KLAnnealing(keras.callbacks.Callback):
    """Linear warmup of both KL weights, preventing posterior collapse."""

    def __init__(self, warmup=15, max_bulk=0.3, max_tail=0.2):
        super().__init__()
        self.warmup, self.max_bulk, self.max_tail = warmup, max_bulk, max_tail

    def on_epoch_begin(self, epoch, logs=None):
        w = min(1.0, (epoch + 1) / self.warmup)
        self.model.kl_weight_bulk = w * self.max_bulk
        self.model.kl_weight_tail = w * self.max_tail


class PiSchedule(keras.callbacks.Callback):
    """Ramp the outage-classification weight once reconstruction has settled."""

    def __init__(self, warmup=5, ramp_end=20, start=0.5, end=5.0):
        super().__init__()
        self.args = (warmup, ramp_end, start, end)

    def on_epoch_begin(self, epoch, logs=None):
        self.model.lambda_pi = _ramp(epoch, *self.args)


class OutageReconSchedule(keras.callbacks.Callback):
    """Ramp the outage-region reconstruction weight."""

    def __init__(self, warmup=3, ramp_end=20, start=1.0, end=10.0):
        super().__init__()
        self.args = (warmup, ramp_end, start, end)

    def on_epoch_begin(self, epoch, logs=None):
        self.model.lambda_outage_recon = _ramp(epoch, *self.args)


class PiSharpSchedule(keras.callbacks.Callback):
    """Ramp the entropy penalty only after classification is established."""

    def __init__(self, warmup=10, ramp_end=35, start=0.1, end=3.0):
        super().__init__()
        self.args = (warmup, ramp_end, start, end)

    def on_epoch_begin(self, epoch, logs=None):
        self.model.lambda_pi_sharp = _ramp(epoch, *self.args)


class OutageMonitor(keras.callbacks.Callback):
    """Every `every` epochs, report outage-region RMSE in dB on a few test maps.

    Training loss alone is a poor proxy for the metric we care about, so this
    surfaces the quantity reported in the paper during the run.
    """

    def __init__(self, X_test, y_test, outage_test, scaler, every=5, n_maps=8):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.outage_test = outage_test
        self.scaler = scaler
        self.every = every
        self.n_maps = n_maps

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0:
            return

        n = min(self.n_maps, len(self.X_test))
        mu, _ls, y_tail, _exc, pi = self.model.generate_maps(self.X_test[:n])
        mu = mu.numpy()[:, :, :, 0]
        y_tail = y_tail.numpy()[:, :, :, 0]
        pi = pi.numpy()[:, :, :, 0]

        valid = self.X_test[:n, :, :, 0] <= 0.5
        is_out = self.outage_test[:n, :, :, 0] > 0.5

        blend = (1.0 - pi) * mu + pi * y_tail
        pred_db = self.scaler.inverse_transform(blend[valid])
        true_db = self.scaler.inverse_transform(self.y_test[:n, :, :, 0][valid])
        out_v = is_out[valid]

        if out_v.sum() > 0:
            rmse_out = float(np.sqrt(np.mean((pred_db[out_v] - true_db[out_v]) ** 2)))
            rmse_good = float(
                np.sqrt(np.mean((pred_db[~out_v] - true_db[~out_v]) ** 2))
            )
            pi_out = float(pi[valid][out_v].mean())
            pi_good = float(pi[valid][~out_v].mean())
        else:  # pragma: no cover
            rmse_out = rmse_good = pi_out = pi_good = float("nan")

        temperature = float(self.model.decoder.temperature.numpy()[0])
        print(
            f"\n  [epoch {epoch + 1}] outage RMSE={rmse_out:.2f} dB  "
            f"bulk RMSE={rmse_good:.2f} dB  "
            f"pi_out={pi_out:.3f} pi_good={pi_good:.4f}  T={temperature:.3f}"
        )


class Progress(keras.callbacks.Callback):
    """Compact per-epoch progress bar (Keras verbose=0 is used instead)."""

    def __init__(self, total_batches: int, total_epochs: int):
        super().__init__()
        self.total_batches = total_batches
        self.total_epochs = total_epochs
        self.epoch_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{self.total_epochs}")

    def on_train_batch_end(self, batch, logs=None):
        done = batch + 1
        filled = int(30 * done / max(self.total_batches, 1))
        elapsed = time.time() - self.epoch_start
        eta = elapsed / done * (self.total_batches - done) if done else 0.0
        bar = "#" * filled + "." * (30 - filled)
        print(
            f"\r  {bar} {done}/{self.total_batches} "
            f"[{elapsed:.0f}s<{eta:.0f}s] loss={logs.get('total', 0):.4f}",
            end="",
        )

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start
        print(f"\n  done in {elapsed:.1f}s  val_loss={logs.get('val_total', 0):.4f}")


def default_callbacks(cfg, X_test, y_test, outage_test, scaler, total_batches):
    kl_warmup, kl_bulk, kl_tail = cfg.kl_schedule
    return [
        KLAnnealing(kl_warmup, kl_bulk, kl_tail),
        PiSchedule(*cfg.pi_schedule),
        OutageReconSchedule(*cfg.outage_recon_schedule),
        PiSharpSchedule(*cfg.pi_sharp_schedule),
        OutageMonitor(X_test, y_test, outage_test, scaler),
        Progress(total_batches, cfg.epochs),
    ]
