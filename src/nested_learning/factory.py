from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .data import SyntheticTextConfig, SyntheticTextDataset, TokenShardDataset, collate_batch
from .levels import LevelSpec
from .model import HOPEModel, ModelConfig
from .optim.m3 import M3
from .titan.model import TitanOnlyModel, TitanOnlyModelConfig


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


def build_model(
    config: DictConfig | dict[str, Any] | ModelConfig | TitanOnlyModelConfig,
) -> torch.nn.Module:
    if is_dataclass(config):
        return _build_model_from_dataclass(config)
    return build_model_from_cfg(config)


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
