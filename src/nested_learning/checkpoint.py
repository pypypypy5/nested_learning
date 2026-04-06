from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        payload = cast(
            dict[str, Any],
            torch.load(path, map_location=map_location, weights_only=True),
        )
    except TypeError:
        payload = cast(dict[str, Any], torch.load(path, map_location=map_location))
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
