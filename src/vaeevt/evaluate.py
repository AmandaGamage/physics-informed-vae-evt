

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Config, set_global_seed
from .dataset import build_dataset, load_radiomapseer, rebuild_for_threshold
from .metrics import evaluate_predictions, flatten_valid, per_map_rmse, select_tau
from .model import build_model


def load_trained_model(run_dir: Path, cfg: Config):
    """Restore a checkpoint written by `vaeevt.train`."""
    with open(run_dir / "evt_params.pkl", "rb") as fh:
        evt_params = pickle.load(fh)
    with open(run_dir / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)

    model = build_model(evt_params, cfg)
    checkpoint = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(str(run_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    checkpoint.restore(latest).expect_partial()
    return model, scaler, evt_params


def select_global_tau(model, X_test, y_test, scaler, cfg, verbose=True):
    """Select the routing threshold t* once, at the training outage quantile.

    A single t* is then reused at every evaluation threshold and in every
    figure, so that all reported numbers describe the same decision rule.

    Re-selecting t* per threshold instead makes the outage RMSE incomparable
    across rows: because t* is chosen by maximising F1, and the outage label
    set changes with the quantile, the optimum swings wildly (0.0030 to 0.9900
    across the published runs). At a low t* the sharpened mask saturates near 1
    almost everywhere, the prediction collapses onto the tail head, and the
    outage RMSE drops for reasons that have nothing to do with model quality.
    """
    if cfg.tau is not None:
        if verbose:
            print(f"\nUsing pinned routing threshold t* = {cfg.tau:.4f}")
        return float(cfg.tau), None

    if verbose:
        print(
            f"\nSelecting t* once at the training quantile "
            f"q = {100 * cfg.target_outage_frac:g}%"
        )

    X_q, outage_q, _ = rebuild_for_threshold(
        X_test, y_test, scaler, cfg.target_outage_frac, seed=cfg.seed
    )
    mu, _ls, y_tail, _exc, pi = model.generate_maps_batched(
        X_q, cfg.inference_batch_size
    )
    flat = flatten_valid(X_q, y_test, outage_q, mu, y_tail, pi, scaler)
    tau, f1 = select_tau(flat["is_outage"], flat["pi"], cfg.tau_grid_size)

    if verbose:
        print(f"  t* = {tau:.4f}  (F1 = {f1:.4f})")
    return tau, f1


def evaluate_at_thresholds(model, X_test, y_test, scaler, cfg, tau=None, verbose=True):
    """Evaluate at every quantile in cfg.eval_thresholds.

    `tau` is the routing threshold. When it is None and
    `cfg.tau_selection == "per_threshold"`, t* is re-derived at each threshold
    (legacy behaviour, not recommended -- see `select_global_tau`).
    """
    results = {}

    for frac in cfg.eval_thresholds:
        label = f"{100 * frac:g}%"
        if verbose:
            print(f"\n--- Outage threshold q = {label} ---")

        X_q, outage_q, thresholds_q = rebuild_for_threshold(
            X_test, y_test, scaler, frac, seed=cfg.seed
        )
        mu, _ls, y_tail, _exc, pi = model.generate_maps_batched(
            X_q, cfg.inference_batch_size
        )

        flat = flatten_valid(X_q, y_test, outage_q, mu, y_tail, pi, scaler)
        metrics = evaluate_predictions(
            flat, cfg.alpha_sharp, cfg.tau_grid_size, tau=tau
        )
        metrics["threshold_frac"] = frac
        metrics["tau_selection"] = "pinned" if tau is not None else "per_threshold"
        metrics["threshold_db_mean"] = float(np.mean(thresholds_q))

        maps = per_map_rmse(
            X_q, y_test, outage_q, mu, y_tail, pi, scaler,
            metrics["tau"], cfg.alpha_sharp,
        )
        metrics["_per_map"] = maps

        if verbose:
            print(
                f"  outage pixels : {metrics['n_outage_pixels']:,} "
                f"({100 * metrics['outage_fraction']:.2f}%)"
            )
            print(
                f"  RMSE  outage={metrics['rmse_outage_sharp']:.2f} dB  "
                f"bulk={metrics['rmse_good_sharp']:.2f} dB  "
                f"overall={metrics['rmse_overall_sharp']:.2f} dB"
            )
            print(
                f"  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  AUC={metrics['roc_auc']:.4f}  "
                f"tau={metrics['tau']:.4f}"
            )

        results[label] = metrics

    return results


def write_reports(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, m in results.items():
        maps = m["_per_map"]
        n = len(maps["rmse_outage_sharp"])
        for i in range(n):
            row = {"threshold": label, "map_index": i}
            row.update({k: v[i] for k, v in maps.items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "per_map_rmse.csv", index=False)

    for label, m in results.items():
        safe = label.replace("%", "pct").replace(".", "_")
        pd.DataFrame({"fpr": m["_fpr"], "tpr": m["_tpr"]}).to_csv(
            out_dir / f"roc_{safe}.csv", index=False
        )

    print(f"\nReports written to {out_dir.resolve()}")
    return


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate Physics-informed VAE-EVT")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--run", type=str, required=True, help="training output dir")
    p.add_argument("--out", type=str, default=None, help="report dir")
    p.add_argument(
        "--tau",
        type=float,
        default=None,
        help="pin the routing threshold t* to this value instead of selecting it",
    )
    p.add_argument(
        "--tau-selection",
        choices=("global", "per_threshold"),
        default=None,
        help="'global' (default): select t* once at the training quantile and "
        "reuse it everywhere. 'per_threshold': re-select at each threshold "
        "(legacy, makes rows incomparable).",
    )
    args = p.parse_args(argv)

    run_dir = Path(args.run)
    cfg = Config.from_json(run_dir / "config.json")
    cfg.radiomapseer_dir = args.data
    if args.tau is not None:
        cfg.tau = args.tau
    if args.tau_selection is not None:
        cfg.tau_selection = args.tau_selection
    set_global_seed(cfg.seed)

    model, scaler, _evt = load_trained_model(run_dir, cfg)

    print("\nRebuilding the test split (same seed as training)")
    samples, all_snr = load_radiomapseer(cfg)
    data = build_dataset(samples, all_snr, cfg)
    _train_idx, test_idx = data.split(cfg.test_size, cfg.seed)
    X_test, y_test = data.X[test_idx], data.y[test_idx]

    tau, tau_f1 = None, None
    if cfg.tau_selection == "global" or cfg.tau is not None:
        tau, tau_f1 = select_global_tau(model, X_test, y_test, scaler, cfg)

    results = evaluate_at_thresholds(model, X_test, y_test, scaler, cfg, tau=tau)

    out_dir = Path(args.out or run_dir / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist t* so that figures generated later reuse the same decision rule.
    with open(out_dir / "tau.json", "w") as fh:
        json.dump(
            {
                "tau": tau,
                "selection_f1": tau_f1,
                "tau_selection": cfg.tau_selection,
                "alpha_sharp": cfg.alpha_sharp,
                "selected_at_quantile": cfg.target_outage_frac,
                "pinned": cfg.tau is not None,
            },
            fh,
            indent=2,
        )

    write_reports(results, out_dir)
    return results


if __name__ == "__main__":
    main()
