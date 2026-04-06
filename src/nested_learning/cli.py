from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import torch
import typer
from omegaconf import OmegaConf

from .config_utils import compose_config
from .device import resolve_device
from .training import (
    build_model_from_cfg,
    generate,
    load_checkpoint,
    run_training_loop,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Minimal Nested Learning CLI.",
)


def _parse_token_ids(tokens: str) -> list[int]:
    values = [part.strip() for part in tokens.split(",")]
    parsed = [int(value) for value in values if value]
    if not parsed:
        raise typer.BadParameter("tokens must contain at least one integer token id")
    return parsed


@app.command("smoke")
def smoke(
    config_name: Annotated[
        str,
        typer.Option("--config-name", "-c", help="Hydra config name."),
    ] = "pilot_smoke",
    device: Annotated[
        str,
        typer.Option("--device", help="Device string such as cpu or cuda:0."),
    ] = "cpu",
    batch_size: Annotated[int, typer.Option("--batch-size")] = 1,
    seq_len: Annotated[int, typer.Option("--seq-len")] = 32,
) -> None:
    cfg = compose_config(config_name)
    torch_device = resolve_device(device)
    model = build_model_from_cfg(cfg.model).to(torch_device)
    model.eval()
    tokens = torch.randint(
        0,
        int(cfg.model.vocab_size),
        (batch_size, seq_len),
        device=torch_device,
    )
    with torch.no_grad():
        logits = model(tokens)
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "config_name": config_name,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "logits_shape": list(cast(torch.Tensor, logits).shape),
            },
            sort_keys=True,
        )
    )


@app.command("train")
def train(
    config_name: Annotated[
        str,
        typer.Option("--config-name", "-c", help="Hydra config name."),
    ] = "pilot",
    override: Annotated[
        list[str] | None,
        typer.Option("--override", "-O", help="Hydra override(s)."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Override cfg.train.device."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    cfg = compose_config(config_name, overrides=override or [])
    if device is not None:
        cfg.train.device = device
    if dry_run:
        typer.echo(OmegaConf.to_yaml(cfg))
        return
    metrics = run_training_loop(cfg, device=resolve_device(str(cfg.train.device)))
    typer.echo(json.dumps(metrics, sort_keys=True))


@app.command("infer")
def infer(
    config_name: Annotated[
        str,
        typer.Option("--config-name", "-c", help="Hydra config name."),
    ] = "pilot",
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="Optional model checkpoint."),
    ] = None,
    tokens: Annotated[
        str,
        typer.Option("--tokens", help="Comma-separated token ids, e.g. 1,2,3"),
    ] = "1,2,3",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 16,
    temperature: Annotated[float, typer.Option("--temperature")] = 1.0,
    top_k: Annotated[int | None, typer.Option("--top-k")] = None,
    device: Annotated[
        str,
        typer.Option("--device", help="Device string such as cpu or cuda:0."),
    ] = "cpu",
) -> None:
    cfg = compose_config(config_name)
    torch_device = resolve_device(device)
    model = build_model_from_cfg(cfg.model).to(torch_device)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location=torch_device)
    prompt = torch.tensor([_parse_token_ids(tokens)], dtype=torch.long, device=torch_device)
    generated = generate(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    typer.echo(json.dumps({"tokens": generated[0].tolist()}, sort_keys=True))
