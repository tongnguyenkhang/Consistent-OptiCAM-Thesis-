# Re-export public API của cam_utils

from .models import (
    get_backbone,       # noqa: F401
    build_model_seq,    # noqa: F401
    count_parameters,   # noqa: F401
    get_preprocess,     # noqa: F401
)
from .train import (
    get_loader,         # noqa: F401
    train_ce,           # noqa: F401
    train_kd_fixed,     # noqa: F401
)
from .eval_utils import (
    ensure_dir,                     # noqa: F401
    sanitize_name,                  # noqa: F401
    compute_metrics_per_sample,     # noqa: F401
    insertion_deletion_auc,         # noqa: F401
    compute_morf_single,            # noqa: F401
    run_opticam,                    # noqa: F401
    eval_and_dump,                  # noqa: F401
)

# Khai báo public API để Pylance/linters không báo "not accessed"
__all__ = [
    # models
    "get_backbone",
    "build_model_seq",
    "count_parameters",
    "get_preprocess",
    # train
    "get_loader",
    "train_ce",
    "train_kd_fixed",
    # eval_utils
    "ensure_dir",
    "sanitize_name",
    "compute_metrics_per_sample",
    "insertion_deletion_auc",
    "compute_morf_single",
    "run_opticam",
    "eval_and_dump",
]