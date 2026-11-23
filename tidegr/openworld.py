"""Open-world priors (v1 Eq. 30-32).

New item i:  mu_i^0 = W t_i, lambda_i^0 = 1 / sigma_res^2 (per dimension).
W = precision-weighted ridge regression of warm item means on item features.
For speed the weight of item i is its mean precision over dimensions (one
solve instead of d solves); stated as an approximation.
New user u: mu_u^0 = zeta * mean of the propagated means of the items in its
first interactions; lambda_u^0 = lambda_0.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def fit_text_map(T: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor, lambda_w: float = 1.0):
    """T [n,f] features, mu [n,d] item means, lam [n,d] precisions -> (W [f,d], var_res [d])."""
    if T.shape[0] < 10:
        return None, None
    w = lam.mean(1)
    w = w / w.mean().clamp_min(1e-12)
    Tw = T * w.unsqueeze(1)
    G = T.T @ Tw + lambda_w * torch.eye(T.shape[1], device=T.device, dtype=T.dtype)
    W = torch.linalg.solve(G, Tw.T @ mu)
    res = mu - T @ W
    var = (w.unsqueeze(1) * res ** 2).sum(0) / w.sum().clamp_min(1e-12)
    return W, var.clamp_min(1e-6)
