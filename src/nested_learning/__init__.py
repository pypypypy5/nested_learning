"""Nested Learning (HOPE) reproduction package."""

from importlib.metadata import PackageNotFoundError, version

from .checkpoint import load_checkpoint, save_checkpoint  # noqa: F401
from .factory import (  # noqa: F401
    build_dataloader,
    build_model,
    build_model_from_cfg,
    build_optimizer,
)
from .inference import generate  # noqa: F401
from .api import (  # noqa: F401
    next_token_loss,
    run_training_loop,
    train_step,
)
from .levels import LevelClock, LevelSpec  # noqa: F401

try:
    __version__ = version("nested-learning")
except PackageNotFoundError:  # pragma: no cover - editable/local source tree
    __version__ = "0.2.0"

__all__ = [
    "LevelClock",
    "LevelSpec",
    "build_dataloader",
    "build_model",
    "build_model_from_cfg",
    "build_optimizer",
    "generate",
    "load_checkpoint",
    "next_token_loss",
    "run_training_loop",
    "save_checkpoint",
    "train_step",
    "__version__",
]
