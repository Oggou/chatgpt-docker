import math
from typing import List, Optional, Sequence, Literal

import torch
import torch.nn.functional as F


PoolMode = Literal["last", "mean"]


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def cos_to_01(cos_sim: torch.Tensor) -> torch.Tensor:
    """
    Map cosine similarity from [-1, 1] -> [0, 1].
    """
    return 0.5 * (cos_sim + 1.0)


def _pool_hidden(hidden_states: Sequence[torch.Tensor], mode: PoolMode = "last") -> torch.Tensor:
    """
    hidden_states: list/tuple of tensors, each (batch, seq_len, hidden_dim)
    Returns: (batch, hidden_dim) pooled across layers.
    """
    if len(hidden_states) == 0:
        raise ValueError("hidden_states must be non-empty")

    pooled_layers: List[torch.Tensor] = []
    for h in hidden_states:
        if h.ndim != 3:
            raise ValueError(f"Expected hidden state with shape (batch, seq, dim); got {tuple(h.shape)}")
        if mode == "last":
            pooled_layers.append(h[:, -1, :])
        elif mode == "mean":
            pooled_layers.append(h.mean(dim=1))
        else:
            raise ValueError(f"Unknown pool mode: {mode}")

    # average across layers to reduce per-layer noise
    return torch.stack(pooled_layers, dim=0).mean(dim=0)  # (batch, hidden_dim)


def semantic_drift_score(
    hidden_t: Sequence[torch.Tensor],
    hidden_prev: Optional[Sequence[torch.Tensor]],
    *,
    pool: PoolMode = "last",
    prior: float = 0.5,
) -> float:
    """
    SDS (local semantic stability) in [0,1].
      1.0 => adjacent hidden representations are very similar (stable trajectory)
      0.0 => adjacent hidden representations are very dissimilar (drift)
    """
    if hidden_prev is None:
        return float(prior)

    h_curr = _pool_hidden(hidden_t, mode=pool)
    h_prev = _pool_hidden(hidden_prev, mode=pool)

    h_curr = F.normalize(h_curr, dim=-1)
    h_prev = F.normalize(h_prev, dim=-1)

    cos = (h_curr * h_prev).sum(dim=-1).mean()  # scalar
    sds01 = cos_to_01(cos)
    sds01 = torch.clamp(sds01, 0.0, 1.0)
    return float(sds01.item())


def context_influence_score(
    hidden_t: Sequence[torch.Tensor],
    context_emb: Optional[torch.Tensor],
    *,
    pool: PoolMode = "last",
    prior: float = 0.5,
    sharpen: bool = True,
) -> float:
    """
    CIS (prompt anchoring) in [0,1].
      1.0 => strongly aligned with context embedding
      0.0 => strongly anti-aligned / unanchored

    context_emb: (batch, hidden_dim)
    """
    if context_emb is None:
        return float(prior)

    h_curr = _pool_hidden(hidden_t, mode=pool)  # (batch, hidden_dim)
    h_curr = F.normalize(h_curr, dim=-1)
    ctx = F.normalize(context_emb, dim=-1)

    cos = (h_curr * ctx).sum(dim=-1).mean()
    cis01 = cos_to_01(cos)
    cis01 = torch.clamp(cis01, 0.0, 1.0)

    if sharpen:
        # mild nonlinearity to increase sensitivity near collapse without
        # creating hard discontinuities.
        cis01 = 0.4 * cis01 + 0.6 * (cis01 ** 2)

    cis01 = torch.clamp(cis01, 0.0, 1.0)
    return float(cis01.item())
