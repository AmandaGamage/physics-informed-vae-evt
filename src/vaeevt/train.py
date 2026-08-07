"""Training entry point.

    python -m vaeevt.train --data /path/to/RadioMapSeer --outage-frac 0.001

Reproduces the training run behind the reported results: 300 layouts x 10
transmitters = 3000 environment-transmitter pairs, split 80/20 into 2400
training and 600 test samples.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .callbacks import default_callbacks
from .config import Config, set_global_seed
from .dataset import build_dataset, load_radiomapseer
from .model import build_model


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Train Physics-informed VAE-EVT")
    p.add_argument("--data", type=str, help="path to the RadioMapSeer root")
    p.add_argument("--config", type=str, help="JSON config to load")
    p.add_argument("--outage-frac", type=float, help="q as a fraction, e.g. 0.001")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--n-maps", type=int)
    p.add_argument("--n-tx", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--out", type=str, help="output directory")
    args = p.parse_args(argv)

    cfg = Config.from_json(args.config) if args.config else Config()
    if args.data:
        cfg.radiomapseer_dir = args.data
    if args.outage_frac is not None:
        cfg.target_outage_frac = args.outage_frac
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.n_maps is not None:
        cfg.n_maps = args.n_maps
    if args.n_tx is not None:
        cfg.n_tx_per_map = args.n_tx
    if args.seed is not None:
        cfg.seed = args.seed
    if args.out:
        cfg.output_dir = args.out
    return cfg


def configure_gpu() -> None:
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:  # pragma: no cover
            pass


def main(argv=None):
    cfg = parse_args(argv)
    set_global_seed(cfg.seed)
    configure_gpu()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out_dir / "config.json")

    print("=" * 70)
    print("Physics-informed VAE-EVT for tail-aware radio map prediction")
    print("=" * 70)
    print(f"TensorFlow {tf.__version__}")
    print(f"Noise floor: {cfg.noise_floor_dbm:.1f} dBm")
    print(f"Outage quantile q = {100 * cfg.target_outage_frac:.3g}%")

    # ------------------------------------------------------------------
    print("\n[1/4] Loading RadioMapSeer and computing physics features")
    samples, all_snr = load_radiomapseer(cfg)

    print("\n[2/4] Building input tensors and fitting GPD anchors")
    data = build_dataset(samples, all_snr, cfg)

    train_idx, test_idx = data.split(cfg.test_size, cfg.seed)
    X_train, X_test = data.X[train_idx], data.X[test_idx]
    y_train, y_test = data.y[train_idx], data.y[test_idx]
    o_train, o_test = data.outage_mask[train_idx], data.outage_mask[test_idx]
    print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")

    # ------------------------------------------------------------------
    print("\n[3/4] Building model")
    model = build_model(data.evt_params, cfg)
    n_enc = model.encoder.count_params()
    n_dec = model.decoder.count_params()
    print(f"  Encoder: {n_enc:,}   Decoder: {n_dec:,}   Total: {n_enc + n_dec:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate, amsgrad=True)
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, (y_train, o_train)))
        .shuffle(500, seed=cfg.seed)
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, (y_test, o_test)))
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ------------------------------------------------------------------
    print(f"\n[4/4] Training for {cfg.epochs} epochs")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=default_callbacks(
            cfg, X_test, y_test, o_test, data.scaler,
            total_batches=len(X_train) // cfg.batch_size,
        )
        + [keras.callbacks.LambdaCallback(on_epoch_end=lambda e, l: __import__("gc").collect())],
        verbose=0,
    )

    # ------------------------------------------------------------------
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.save(str(out_dir / "ckpt"))
    with open(out_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(data.scaler, fh)
    with open(out_dir / "evt_params.pkl", "wb") as fh:
        pickle.dump(data.evt_params, fh)
    np.savez_compressed(
        out_dir / "split.npz", train_idx=train_idx, test_idx=test_idx
    )
    with open(out_dir / "history.json", "w") as fh:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, fh, indent=2)

    print(f"\nArtifacts written to {out_dir.resolve()}")
    return model, data, (train_idx, test_idx)


if __name__ == "__main__":
    main()
