# Nested Learning Research Core

Lightweight research repo for experimenting with the model mechanisms in Nested Learning / HOPE with a minimal surface: build model, train, save/load, infer.

The model-side implementation is intentionally preserved:
- HOPE block variants
- CMS
- Self-Modifying TITANs
- fast-state and online update paths
- local train and inference API

## What Remains

- Core model code under `src/nested_learning/`
- Minimal public API under `nested_learning.api`
- Local entrypoints: `train.py`, `nl train`, `nl infer`, `nl smoke`
- A small config surface:
  - `configs/pilot.yaml`
  - `configs/pilot_smoke.yaml`
  - `configs/hope/pilot.yaml`
  - `configs/hope/pilot_attention.yaml`
  - `configs/hope/pilot_selfmod.yaml`
  - `configs/hope/pilot_transformer.yaml`
- A reduced test suite centered on model behavior

## Quickstart

```bash
uv python install 3.12
uv sync --all-extras
uv run nl smoke --config-name pilot_smoke --device cpu
uv run python train.py --config-name pilot_smoke
uv run nl infer --config-name pilot --tokens 1,2,3 --max-new-tokens 8
uv run pytest
```

## CLI

```bash
uv run nl smoke --config-name pilot_smoke --device cpu
uv run nl train --config-name pilot --override train.steps=100 --override train.device=cpu
uv run nl infer --config-name pilot --tokens 1,2,3 --max-new-tokens 16
```

`python -m nested_learning ...` is also supported.

## Python API

```python
import torch
from nested_learning import build_model_from_cfg, build_optimizer, generate, train_step
from nested_learning.config_utils import compose_config

cfg = compose_config("pilot")
model = build_model_from_cfg(cfg.model)
optimizer = build_optimizer(model, cfg.optim)

tokens = torch.randint(0, cfg.model.vocab_size, (2, 64))
metrics = train_step(model, optimizer, tokens, device=torch.device("cpu"))

prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
out = generate(model, prompt, max_new_tokens=8)
```

## Notes

- Distributed launchers, eval harnesses, release automation, and project-management layers were removed.
- `training.py` is now a small local train loop rather than a full experiment orchestrator.
- The remaining configs are only there to build and run the models easily while you modify architecture code directly.
- High-level module structure and development rules are documented in `docs/PROJECT.md` and `docs/DEVELOPMENT.md`.
