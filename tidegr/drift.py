"""Evidence-vs-change drift detector (v1 Eq. 25-29).

Standardised predictive surprise per node:
  l_j = BCE(y_j, p_j),  E0[l_j] = H(p_j),  Var0[l_j] = p(1-p) (log(p/(1-p)))^2
  S_v = sum_j (l_j - H(p_j)) / sqrt(sum_j Var0[l_j] + eps)   ~ N(0,1) under H0
One-sided CUSUM: C_v <- max(0, C_v + S_v - delta) (active), C_v <- varsigma*C_v (inactive).
Ablation "raw": S_v = z-score of the node's mean loss among active nodes.
"""
from __future__ import annotations

import numpy as np


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    p = a / max(a.sum(), 1e-12)
    q = b / max(b.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(x, y):
        nz = x > 0
        return float((x[nz] * np.log(x[nz] / y[nz])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class DriftDetector:
    def __init__(self, n_nodes: int, mode: str = "surprise", delta: float = 0.5,
                 varsigma: float = 0.9, eps: float = 1e-6, delta_g: float = 0.5, h_g: float = 3.0):
        self.mode = mode
        self.delta, self.varsigma, self.eps = float(delta), float(varsigma), float(eps)
        self.delta_g, self.h_g = float(delta_g), float(h_g)
        self.C = np.zeros(n_nodes, np.float64)
        self.last_S = np.zeros(n_nodes, np.float64)
        self.G = 0.0
        self.js_warm: list = []
        self.alarm = False
        self.last_js = 0.0
        self.last_mean_S = 0.0

    @property
    def D(self) -> np.ndarray:
        return self.C

    def _per_node(self, node: np.ndarray, y: np.ndarray, p: np.ndarray, n: int) -> np.ndarray:
        p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
        y = y.astype(np.float64)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        if self.mode == "raw":
            cnt = np.bincount(node, minlength=n)
            mean_l = np.bincount(node, weights=loss, minlength=n) / np.maximum(cnt, 1)
            act = cnt > 0
            mu, sd = mean_l[act].mean(), mean_l[act].std() + 1e-8
            S = np.zeros(n)
            S[act] = (mean_l[act] - mu) / sd
            return S
        H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        var0 = p * (1 - p) * np.log(p / (1 - p)) ** 2
        num = np.bincount(node, weights=loss - H, minlength=n)
        den = np.bincount(node, weights=var0, minlength=n)
        return num / np.sqrt(den + self.eps)

    def update(self, users: np.ndarray, items_node: np.ndarray, y: np.ndarray, p: np.ndarray,
               active_nodes: np.ndarray, pop_prev: np.ndarray | None = None,
               pop_new: np.ndarray | None = None, warmup: bool = False):
        n = len(self.C)
        S_u = self._per_node(users, y, p, n)
        S_i = self._per_node(items_node, y, p, n)
        S = S_u + S_i  # users and items occupy disjoint index ranges
        act = np.zeros(n, bool)
        act[active_nodes] = True
        self.C[act] = np.maximum(0.0, self.C[act] + S[act] - self.delta)
        self.C[~act] *= self.varsigma
        self.last_S = S
        ua = np.unique(users)
        self.last_mean_S = float(S[ua].mean()) if len(ua) else 0.0
        self.G = max(0.0, self.G + self.last_mean_S - self.delta_g)
        js = js_divergence(pop_prev, pop_new) if (pop_prev is not None and pop_new is not None) else 0.0
        self.last_js = js
        if warmup:
            self.js_warm.append(js)
            self.alarm = False
        else:
            q95 = np.quantile(self.js_warm, 0.95) if len(self.js_warm) >= 3 else np.inf
            self.alarm = bool(self.G > self.h_g or js > q95)
        return S

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, d):
        self.__dict__.update(d)
