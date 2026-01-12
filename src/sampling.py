from __future__ import annotations

from typing import Optional, Tuple

import torch


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # subtract max for numerical stability
    z = logits - logits.max(dim=dim, keepdim=True).values
    return torch.softmax(z, dim=dim)


def mix_distributions(p: torch.Tensor, q: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Returns (1-lam)*p + lam*q, renormalized.
    """
    lam = float(lam)
    lam = 0.0 if lam < 0.0 else (1.0 if lam > 1.0 else lam)
    r = (1.0 - lam) * p + lam * q
    r = r / r.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return r


def trajectory_regularized_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    prev_probs: Optional[torch.Tensor] = None,
    mix_lambda: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply temperature and (optionally) a KL-leash-style distribution mix vs prev_probs.

    Returns:
      adjusted_logits: log(p_adjusted)
      probs: p_adjusted (for caching)
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape (batch, vocab); got {tuple(logits.shape)}")

    T = float(temperature)
    if not (T > 0.0):
        T = 1.0

    p = stable_softmax(logits / T, dim=-1)

    if mix_lambda and prev_probs is not None:
        p = mix_distributions(p, prev_probs, mix_lambda)

    adjusted_logits = torch.log(p.clamp_min(1e-12))
    return adjusted_logits, p
