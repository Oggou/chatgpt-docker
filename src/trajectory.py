from typing import Optional, Sequence, Literal

import torch
import torch.nn.functional as F

from .metrics import cos_to_01


PoolMode = Literal["last", "mean"]


def _pool_hidden(hidden_states: Sequence[torch.Tensor], mode: PoolMode = "last") -> torch.Tensor:
    if len(hidden_states) == 0:
        raise ValueError("hidden_states must be non-empty")

    pooled_layers = []
    for h in hidden_states:
        if h.ndim != 3:
            raise ValueError(f"Expected hidden state with shape (batch, seq, dim); got {tuple(h.shape)}")
        if mode == "last":
            pooled_layers.append(h[:, -1, :])
        elif mode == "mean":
            pooled_layers.append(h.mean(dim=1))
        else:
            raise ValueError(f"Unknown pool mode: {mode}")

    return torch.stack(pooled_layers, dim=0).mean(dim=0)  # (batch, dim)


def trajectory_curvature(
    hidden_t: Sequence[torch.Tensor],
    hidden_prev: Optional[Sequence[torch.Tensor]],
    hidden_prev_prev: Optional[Sequence[torch.Tensor]],
    *,
    pool: PoolMode = "last",
    prior: float = 0.5,
) -> float:
    """
    TC (trajectory curvature proxy) in [0,1].
      0.0 => same semantic direction (smooth)
      1.0 => sharp turn / reversal

    Computed as curvature = 1 - cos01( cos( v_t, v_{t-1} ) )
      where v_t = h_t - h_{t-1}
    """
    if hidden_prev is None or hidden_prev_prev is None:
        return float(prior)

    h_curr = _pool_hidden(hidden_t, mode=pool)
    h_prev = _pool_hidden(hidden_prev, mode=pool)
    h_prev_prev = _pool_hidden(hidden_prev_prev, mode=pool)

    v1 = h_curr - h_prev
    v0 = h_prev - h_prev_prev

    v1 = F.normalize(v1, dim=-1)
    v0 = F.normalize(v0, dim=-1)

    cos = (v1 * v0).sum(dim=-1).mean()
    direction01 = cos_to_01(cos)  # [0,1]
    direction01 = torch.clamp(direction01, 0.0, 1.0)

    curvature01 = 1.0 - direction01
    curvature01 = torch.clamp(curvature01, 0.0, 1.0)
    return float(curvature01.item())
