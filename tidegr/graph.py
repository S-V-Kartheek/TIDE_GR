"""Cumulative user-item graph and LightGCN propagation operators.

Nodes: users 0..n_users-1, items n_users..n_users+n_items-1.
Edge weight r_ui = rho(age) = rho_inf + (rho_0 - rho_inf) * exp(-age / tau_rho)
(v1 Eq. 3), age in blocks since the last interaction on that edge.
A_hat(r)_ui = r_ui / sqrt((D_u + eps)(D_i + eps)),  D_u = sum_i r_ui     (v1 Eq. 6)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GraphState:
    def __init__(self, n_users: int, n_items: int):
        self.n_users, self.n_items = int(n_users), int(n_items)
        self.N = self.n_users + self.n_items
        self.key = np.zeros(0, dtype=np.int64)  # u * n_items + i, sorted
        self.last = np.zeros(0, dtype=np.int64)
        self.count = np.zeros(0, dtype=np.int64)
        self.version = 0
        self._cache: dict = {}

    # ------------------------------------------------------------------ edges
    @property
    def n_edges(self) -> int:
        return len(self.key)

    def edges(self):
        return self.key // self.n_items, self.key % self.n_items

    def add(self, u: np.ndarray, i: np.ndarray, block: int):
        if len(u) == 0:
            return
        k_new = np.unique(np.asarray(u, np.int64) * self.n_items + np.asarray(i, np.int64))
        allk = np.concatenate([self.key, k_new])
        alll = np.concatenate([self.last, np.full(len(k_new), block, np.int64)])
        allc = np.concatenate([self.count, np.ones(len(k_new), np.int64)])
        uk, inv = np.unique(allk, return_inverse=True)
        last = np.full(len(uk), -1, np.int64)
        np.maximum.at(last, inv, alll)
        cnt = np.zeros(len(uk), np.int64)
        np.add.at(cnt, inv, allc)
        self.key, self.last, self.count = uk, last, cnt
        self.version += 1
        self._cache.clear()

    def degree_users(self) -> np.ndarray:
        u, _ = self.edges()
        return np.bincount(u, minlength=self.n_users)

    def degree_items(self) -> np.ndarray:
        _, i = self.edges()
        return np.bincount(i, minlength=self.n_items)

    def weights(self, cur_block: int, rho0: float, rhoinf: float, tau: float) -> np.ndarray:
        age = np.maximum(cur_block - self.last, 0).astype(np.float64)
        if tau <= 0:
            return np.ones(len(age), np.float32)
        return (rhoinf + (rho0 - rhoinf) * np.exp(-age / tau)).astype(np.float32)

    # ------------------------------------------------------------------ operators
    def adj(self, w: np.ndarray, device, cache_key=None):
        """Return (A_hat, A_hat^{o2}) as coalesced sparse COO tensors on device."""
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]
        u, i = self.edges()
        w = np.asarray(w, np.float64)
        du = np.bincount(u, weights=w, minlength=self.n_users) + 1e-8
        di = np.bincount(i, weights=w, minlength=self.n_items) + 1e-8
        val = w / np.sqrt(du[u] * di[i])
        rows = np.concatenate([u, i + self.n_users])
        cols = np.concatenate([i + self.n_users, u])
        vals = np.concatenate([val, val]).astype(np.float32)
        idx = torch.from_numpy(np.stack([rows, cols]))
        v = torch.from_numpy(vals)
        A = torch.sparse_coo_tensor(idx, v, (self.N, self.N)).coalesce().to(device)
        A2 = torch.sparse_coo_tensor(idx, v * v, (self.N, self.N)).coalesce().to(device)
        if cache_key is not None:
            self._cache[cache_key] = (A, A2)
        return A, A2

    def state_dict(self) -> dict:
        return {"key": self.key, "last": self.last, "count": self.count, "version": self.version}

    def load_state_dict(self, st: dict):
        self.key, self.last, self.count = st["key"], st["last"], st["count"]
        self.version = st["version"]
        self._cache.clear()


def propagate(A, X: torch.Tensor, L: int, noise_eps: float = 0.0, cl_layer: int = 1):
    """LightGCN mean of layers 0..L. With noise_eps>0 adds XSimGCL uniform noise
    after every propagation; also returns the layer used for the contrastive view."""
    cur = X
    acc = X
    cl_view = X
    for l in range(1, L + 1):
        cur = torch.sparse.mm(A, cur)
        if noise_eps > 0:
            noise = F.normalize(torch.rand_like(cur), dim=-1)
            cur = cur + torch.sign(cur) * noise * noise_eps
        acc = acc + cur
        if l == cl_layer:
            cl_view = cur
    return acc / (L + 1), cl_view


def propagate_sq(A2, X: torch.Tensor, L: int) -> torch.Tensor:
    """Variance pass  sum_l c^2 (A^{o2})^l X,  c = 1/(L+1)  (approximation A2)."""
    c2 = 1.0 / (L + 1) ** 2
    cur = X
    acc = X
    for _ in range(L):
        cur = torch.sparse.mm(A2, cur)
        acc = acc + cur
    return acc * c2
