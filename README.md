# Physics-informed VAE-EVT for Tail-Aware Radio Map Prediction

Reference implementation for the paper *Physics-informed VAE-EVT for Tail Aware
Radio Map Prediction* (Gamage, Mehrnia and Gross, KTH Royal Institute of
Technology).

The framework predicts a full 256×256 SNR map in a single forward pass and
models the bulk and the extreme lower tail of the SNR distribution separately:
a Gaussian bulk latent and a GPD-anchored tail latent, routed per pixel by a
learned outage probability. Deterministic scene geometry, line-of-sight,
shadowing, penetration depth, distance is computed up front and fed to the
network as a ten-channel input tensor.


---

## Install

```bash
git clone https://github.com/USERNAME/physics-informed-vae-evt.git
cd physics-informed-vae-evt

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python ≥ 3.10 and TensorFlow 2.19.

**Resources.** A GPU with ≥ 16 GB of memory is recommended. Peak host RAM for
the full run is roughly 20 GB: `X` alone is 3000 × 256 × 256 × 10 float32 ≈
7.9 GB, and both the raw feature dictionaries and the train/test copies of `X`
are briefly live alongside it. Reduce `--n-maps` if memory is tight — the
pipeline scales linearly.



We do not recommend `tensorflow-metal` for this workload: it is version-fragile
against TF 2.19 and is [reported to crash when PyArrow is present in the same
environment](https://developer.apple.com/forums/thread/803658), which `pandas`
may pull in.



## Data

Download [RadioMapSeer](https://radiomapseer.github.io/).

The dataset is not redistributed here and carries its own licence.



## Repository layout

```
src/vaeevt/
├── config.py       every hyperparameter in one dataclass
├── snr.py          link budget, gain → SNR, the [0,1] target scaler
├── outage.py       per-map outage labelling, GPD anchor fitting
├── features.py     ray tracing and the 10-channel physics tensor  (Sec. III-A)
├── dataset.py      RadioMapSeer loading, tensor assembly, re-thresholding
├── layers.py       dual-latent encoder, attention U-Net decoder  (Sec. III-B/C/D)
├── losses.py       the composite objective                        (Sec. III-E)
├── model.py        GPD reparameterisation, training/inference loop
├── callbacks.py    KL warmup and loss-weight ramps                (Sec. IV-A2)
├── metrics.py      outage RMSE, F1/precision/recall, routing threshold
├── evaluate.py     Table I and Fig. 2 reproduction
├── visualize.py    Fig. 2/3/4 plotting
└── train.py        training entry point
```

## The ten input channels

| # | Symbol | Channel | Description |
|---|---|---|---|
| 0 | `B` | `building_map` | binary occupancy |
| 1 | `T_x` | `tx_map` | truncated Gaussian at the transmitter |
| 2 | `M_LOS` | `los_mask` | 1 where the ray to `p_tx` is unobstructed |
| 3 | `d_L` | `los_dist` | LoS-masked normalised log distance, Eq. (5) |
| 4 | `S_NLOS` | `nlos_shadow` | localised shadowing score |
| 5 | `D_NLOS` | `nlos_depth` | normalised penetration depth, Eq. (6) |
| 6 | `E` | `shadow_edge` | LoS/NLoS boundary map, guides spatial attention |
| 7 | `P_outage` | `outage_prior` | coarse geometric outage risk, Eq. (7) |
| 8 | `D_all` | `dist_all` | normalised distance on all free-space pixels |
| 9 | `γ̂_th` | `threshold` | broadcast normalised per-map outage threshold |

Channel 9 is what lets one trained model be evaluated at several outage
quantiles without retraining: rewrite channel 9, rebuild the labels, re-run
inference. That is exactly what `dataset.rebuild_for_threshold` does.

## Using the pieces separately

The physics preprocessing has no TensorFlow dependency and is useful on its own:

```python
from vaeevt import Config, compute_geometric_features, stack_input_tensor

cfg = Config()
feats = compute_geometric_features(tx_y=128, tx_x=64, building_map=B, cfg=cfg)
X = stack_input_tensor(feats, threshold_norm=0.05)   # (256, 256, 10)
```


The code calls on the global np.random.shuffle, so the labels depended on call 
order and differed between the training script and the evaluation script. 
This implementation threads an explicitly seeded np.random.Generator through 
compute_permap_outage_mask, which makes runs reproducible. Numbers will therefore 
not bit-match the original run even though the method is identical.

## Citation

```bibtex
@inproceedings{gamage2026vaeevt,
  title     = {Physics-informed {VAE-EVT} for Tail Aware Radio Map Prediction},
  author    = {Gamage, Amanda Sheron and Mehrnia, Niloofar and Gross, James},
  booktitle = {TODO},
  year      = {2026}
}
```

## Licence

MIT for the code. RadioMapSeer is distributed under its own terms.
