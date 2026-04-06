from __future__ import annotations

from typing import cast

import torch


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
    with torch.inference_mode():
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
