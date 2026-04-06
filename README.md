# Nested Learning Research Core

Lightweight research repo for experimenting with the model mechanisms in Nested Learning / HOPE without the large evaluation, compliance, release, and scaling layers that were previously bundled around it.

The model-side implementation is intentionally preserved:
- HOPE block variants
- CMS
- Self-Modifying TITANs
- fast-state and online update paths
- local training loop and CLI

## What Remains

- Core model code under `src/nested_learning/`
- Local training entrypoints: `train.py`, `nl train`, `nl smoke`, `nl audit`
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
uv run pytest
```

## CLI

```bash
uv run nl doctor --json
uv run nl smoke --config-name pilot_smoke --device cpu
uv run nl audit --config-name pilot
uv run nl train --config-name pilot --override train.steps=100 --override train.device=cpu
```

`python -m nested_learning ...` is also supported.

## Notes

- This repo is now biased toward local architecture research rather than paper-faithful reproduction workflows.
- Distributed launchers, large eval harnesses, release automation, and long-form project docs were removed to reduce surface area.
- The remaining `pilot` configs are a starting point only. You said you will modify the architecture directly, so the codebase is now arranged to make that easier.
