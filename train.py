from __future__ import annotations

import hydra
from omegaconf import DictConfig

from nested_learning.device import resolve_device
from nested_learning.factory import unwrap_config
from nested_learning.trainer import run_training_loop


@hydra.main(config_path="configs", config_name="pilot", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    device = resolve_device(cfg.train.device)
    run_training_loop(cfg, device=device)


if __name__ == "__main__":
    main()
