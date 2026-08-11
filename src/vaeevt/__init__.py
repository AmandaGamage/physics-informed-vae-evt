

from .config import Config, set_global_seed
from .dataset import Dataset, build_dataset, load_radiomapseer, rebuild_for_threshold
from .features import CHANNEL_NAMES, compute_geometric_features, stack_input_tensor
from .metrics import blend, evaluate_predictions, per_map_rmse, select_tau, sharpen
from .outage import compute_permap_outage_mask, fit_gpd_anchors
from .snr import SNRScaler, convert_gain_to_snr

__version__ = "1.0.0"

__all__ = [
    "Config",
    "set_global_seed",
    "Dataset",
    "load_radiomapseer",
    "build_dataset",
    "rebuild_for_threshold",
    "CHANNEL_NAMES",
    "compute_geometric_features",
    "stack_input_tensor",
    "compute_permap_outage_mask",
    "fit_gpd_anchors",
    "SNRScaler",
    "convert_gain_to_snr",
    "evaluate_predictions",
    "per_map_rmse",
    "select_tau",
    "sharpen",
    "blend",
]


def __getattr__(name):
    """Lazily expose the TensorFlow-dependent symbols so that importing the
    package for feature computation alone does not require TensorFlow."""
    if name in {"build_model", "PhysicsInformedVAEEVT"}:
        from . import model as _model

        return getattr(_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
