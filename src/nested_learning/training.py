from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .data import SyntheticTextConfig, SyntheticTextDataset, TokenShardDataset, collate_batch
from .levels import LevelSpec
from .model import HOPEModel, ModelConfig
from .optim.m3 import M3
from .titan.model import TitanOnlyModel, TitanOnlyModelConfig


class _HasLMHead(Protocol):
    lm_head: torch.nn.Linear


def unwrap_config(cfg: DictConfig) -> DictConfig:
    if "model" in cfg:
        return cfg
    if "hope" in cfg:
        return cast(DictConfig, cfg.hope)
    return cfg


def build_model_from_cfg(model_cfg: DictConfig | dict[str, Any]) -> torch.nn.Module:
    if not isinstance(model_cfg, DictConfig):
        model_cfg = cast(DictConfig, OmegaConf.create(model_cfg))
    model_type = str(model_cfg.get("type", "hope"))
    optimizer_cfg = _to_plain_dict(model_cfg.get("optimizers"))
    teach_schedule = _to_plain_dict(model_cfg.get("teach_schedule"))
    qk_l2_norm = bool(model_cfg.get("qk_l2_norm", False))
    local_conv_window = _maybe_int(model_cfg.get("local_conv_window"))
    surprise_threshold = _maybe_float(model_cfg.get("surprise_threshold"))
    surprise_metric = str(model_cfg.get("surprise_metric", "l2"))
    cms_use_layernorm = bool(model_cfg.get("cms_use_layernorm", True))

    if model_type == "titan":
        titan_cfg = TitanOnlyModelConfig(
            vocab_size=int(model_cfg.vocab_size),
            dim=int(model_cfg.dim),
            num_layers=int(model_cfg.num_layers),
            heads=int(model_cfg.heads),
            titan_level=LevelSpec(**_to_plain_dict(model_cfg.titan_level)),
            optimizers=optimizer_cfg,
            teach_scale=float(model_cfg.get("teach_scale", 1.0)),
            teach_clip=float(model_cfg.get("teach_clip", 0.0)),
            teach_schedule=teach_schedule,
            qk_l2_norm=qk_l2_norm,
            local_conv_window=local_conv_window,
            surprise_threshold=surprise_threshold,
            surprise_metric=surprise_metric,
            freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
            self_mod_lr=float(model_cfg.get("self_mod_lr", 1e-3)),
            self_mod_hidden=int(model_cfg.get("self_mod_hidden", 4)),
        )
        return TitanOnlyModel(titan_cfg)

    hope_cfg = ModelConfig(
        vocab_size=int(model_cfg.vocab_size),
        dim=int(model_cfg.dim),
        num_layers=int(model_cfg.num_layers),
        heads=int(model_cfg.heads),
        titan_level=LevelSpec(**_to_plain_dict(model_cfg.titan_level)),
        cms_levels=[LevelSpec(**entry) for entry in _to_plain_list(model_cfg.cms_levels)],
        cms_flush_partial_at_end=bool(model_cfg.get("cms_flush_partial_at_end", False)),
        cms_use_layernorm=cms_use_layernorm,
        optimizers=optimizer_cfg,
        teach_scale=float(model_cfg.get("teach_scale", 1.0)),
        teach_clip=float(model_cfg.get("teach_clip", 0.0)),
        teach_schedule=teach_schedule,
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
        surprise_threshold=surprise_threshold,
        surprise_metric=surprise_metric,
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        qk_l2_norm=qk_l2_norm,
        local_conv_window=local_conv_window,
        self_mod_lr=float(model_cfg.get("self_mod_lr", 1e-3)),
        self_mod_hidden=int(model_cfg.get("self_mod_hidden", 4)),
        self_mod_chunk_size=int(model_cfg.get("self_mod_chunk_size", 1)),
        self_mod_chunk_size_memory=_maybe_int(model_cfg.get("self_mod_chunk_size_memory")),
        self_mod_objective=str(model_cfg.get("self_mod_objective", "l2")),
        self_mod_stopgrad_vhat=bool(model_cfg.get("self_mod_stopgrad_vhat", True)),
        self_mod_use_rank1_precond=bool(model_cfg.get("self_mod_use_rank1_precond", True)),
        self_mod_use_alpha=bool(model_cfg.get("self_mod_use_alpha", True)),
        self_mod_use_skip=bool(model_cfg.get("self_mod_use_skip", True)),
        self_mod_momentum=float(model_cfg.get("self_mod_momentum", 0.0)),
        self_mod_adaptive_q=bool(model_cfg.get("self_mod_adaptive_q", False)),
        self_mod_local_conv_window=_maybe_int(model_cfg.get("self_mod_local_conv_window", 4)),
        transformer_mlp_hidden_multiplier=int(
            model_cfg.get("transformer_mlp_hidden_multiplier", 4)
        ),
        transformer_activation=str(model_cfg.get("transformer_activation", "gelu")),
        block_variant=str(model_cfg.get("block_variant", "hope_hybrid")),
    )
    return HOPEModel(hope_cfg)


def build_optimizer(
    model: torch.nn.Module,
    optim_cfg: DictConfig | dict[str, Any] | None = None,
    *,
    device: torch.device | None = None,
) -> torch.optim.Optimizer:
    cfg = cast(DictConfig, OmegaConf.create(optim_cfg or {}))
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found on model")

    optim_type = str(cfg.get("type", "adamw")).lower()
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    fused_cfg = cfg.get("fused", "auto")
    use_fused = False
    if fused_cfg == "auto":
        use_fused = bool(device is not None and device.type == "cuda" and torch.cuda.is_available())
    else:
        use_fused = bool(fused_cfg)

    if optim_type == "adamw":
        kwargs: dict[str, Any] = {
            "lr": lr,
            "betas": tuple(cfg.get("betas", (0.9, 0.999))),
            "weight_decay": weight_decay,
        }
        if use_fused:
            kwargs["fused"] = True
        return torch.optim.AdamW(params, **kwargs)

    if optim_type == "m3":
        return M3(
            params,
            lr=lr,
            beta1=float(cfg.get("beta1", 0.9)),
            beta2=float(cfg.get("beta2", 0.999)),
            beta3=float(cfg.get("beta3", 0.9)),
            alpha=float(cfg.get("alpha", 1.0)),
            eps=float(cfg.get("eps", 1e-8)),
            ns_steps=int(cfg.get("ns_steps", 3)),
            slow_chunk=int(cfg.get("slow_chunk", 100)),
            weight_decay=weight_decay,
        )

    if optim_type == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("torch.optim.Muon is not available in this PyTorch build")
        muon_cls = getattr(torch.optim, "Muon")
        return muon_cls(  # type: ignore[misc]
            params,
            lr=lr,
            momentum=float(cfg.get("momentum", 0.95)),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer type {optim_type!r}")


def build_dataloader(data_cfg: DictConfig | dict[str, Any]) -> DataLoader:
    cfg = cast(DictConfig, OmegaConf.create(data_cfg))
    source = str(cfg.get("source", "synthetic"))
    if source == "synthetic":
        dataset = SyntheticTextDataset(
            SyntheticTextConfig(
                vocab_size=int(cfg.vocab_size),
                seq_len=int(cfg.seq_len),
                dataset_size=int(cfg.dataset_size),
            )
        )
    elif source == "shards":
        dataset = TokenShardDataset(cfg.shards_dir)
    else:
        raise ValueError(
            f"Unsupported data source {source!r}. Supported sources: ['synthetic', 'shards']"
        )
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=source != "shards",
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_batch,
    )


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


def next_token_loss(model: torch.nn.Module, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def generate(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    out = tokens.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = cast(torch.Tensor, model(out))
            next_logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / temperature
                if top_k is not None and top_k > 0 and top_k < scaled.size(-1):
                    values, indices = torch.topk(scaled, k=top_k, dim=-1)
                    filtered = torch.full_like(scaled, float("-inf"))
                    filtered.scatter_(1, indices, values)
                    scaled = filtered
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
            if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
                break
    return out


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
    payload = cast(dict[str, Any], torch.load(path, map_location=map_location))
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def build_model(config: DictConfig | dict[str, Any] | ModelConfig | TitanOnlyModelConfig) -> torch.nn.Module:
    if is_dataclass(config):
        return _build_model_from_dataclass(config)
    return build_model_from_cfg(config)


def _build_model_from_dataclass(config: object) -> torch.nn.Module:
    if isinstance(config, ModelConfig):
        return HOPEModel(config)
    if isinstance(config, TitanOnlyModelConfig):
        return TitanOnlyModel(config)
    raise TypeError(f"Unsupported dataclass config type: {type(config)!r}")


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return cast(dict[str, Any], OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return cast(dict[str, Any], asdict(value))
    return dict(value)


def _to_plain_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, DictConfig):
        return cast(list[dict[str, Any]], OmegaConf.to_container(value, resolve=True))
    return list(value)


def _maybe_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)
