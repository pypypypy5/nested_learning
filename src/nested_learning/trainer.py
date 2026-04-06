from __future__ import annotations

from typing import Iterator, Protocol, cast

import torch
from omegaconf import DictConfig

from .checkpoint import save_checkpoint
from .factory import build_dataloader, build_model_from_cfg, build_optimizer, unwrap_config


class _HasLMHead(Protocol):
    lm_head: torch.nn.Linear


def compute_teach_signal(
    model: _HasLMHead,
    logits: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    logits_detached = logits.detach()
    probs = torch.softmax(logits_detached, dim=-1)
    residual = probs.clone()
    batch_size, seq_len, _ = residual.shape
    targets = torch.zeros(batch_size, seq_len, device=tokens.device, dtype=tokens.dtype)
    active = torch.zeros(batch_size, seq_len, device=tokens.device, dtype=torch.bool)
    if seq_len > 1:
        targets[:, :-1] = tokens[:, 1:]
        active[:, :-1] = True
    active_f = active.to(dtype=residual.dtype)
    residual.mul_(active_f.unsqueeze(-1))
    safe_targets = torch.where(active, targets, torch.zeros_like(targets))
    residual.scatter_add_(-1, safe_targets.unsqueeze(-1), -active_f.unsqueeze(-1))
    residual = residual / active_f.sum().clamp(min=1.0)
    head_weight = model.lm_head.weight.detach()
    if head_weight.dtype != residual.dtype:
        head_weight = head_weight.to(dtype=residual.dtype)
    return residual @ head_weight


def next_token_loss(
    model: torch.nn.Module,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = cast(torch.Tensor, model(tokens))
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    )
    return loss, logits


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    clip_grad_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    batch = tokens.to(device)
    optimizer.zero_grad()
    loss, logits = next_token_loss(model, batch)
    loss.backward()
    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
    optimizer.step()
    with torch.no_grad():
        teach_signal = compute_teach_signal(cast(_HasLMHead, model), logits, batch)
        cast(torch.nn.Module, model)(batch, teach_signal=teach_signal)
    ppl = float(torch.exp(loss.detach()).item())
    teach_norm = float(teach_signal.norm(dim=-1).mean().item())
    return {"loss": float(loss.item()), "ppl": ppl, "teach_signal_norm": teach_norm}


def run_training_loop(
    cfg: DictConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.get("optim"), device=device)
    dataloader = build_dataloader(cfg.data)
    steps = int(cfg.train.get("steps", 100))
    log_interval = max(int(cfg.train.get("log_interval", 10)), 1)
    save_path = cfg.train.get("save_path")
    step_iter: Iterator[torch.Tensor] = iter(dataloader)
    latest_metrics = {"loss": 0.0, "ppl": 0.0, "teach_signal_norm": 0.0}

    for step in range(steps):
        try:
            tokens = next(step_iter)
        except StopIteration:
            step_iter = iter(dataloader)
            tokens = next(step_iter)
        latest_metrics = train_step(model, optimizer, tokens, device=device)
        if step % log_interval == 0:
            print(
                f"[train] step={step} loss={latest_metrics['loss']:.4f} "
                f"ppl={latest_metrics['ppl']:.2f} "
                f"teach_norm={latest_metrics['teach_signal_norm']:.4f}"
            )

    if save_path:
        save_checkpoint(model, optimizer, save_path)
    return latest_metrics
