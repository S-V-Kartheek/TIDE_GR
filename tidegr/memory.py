"""Environment-stratified replay memory with seasonal slots (v1 §11, Eq. 36).

* Each environment (block) keeps a slice of its training edges chosen by
  weighted reservoir sampling (A-Res) with priority
  rho_ui = pi_ui * exp(-D_u/h) * exp(-D_i/h).
* Slices of the most recent ``window`` environments are kept, plus one slice
  per season value (the most recent older environment of that season).
* Evicted slices are returned so their curvature can be handed back to the
  Gaussian belief (each datum counted exactly once).
* A small validation slice per environment (from that block's validation
  split, never trained on) measures forgetting for the RL reward.
"""
from __future__ import annotations

import numpy as np


class ReplayMemory:
    def __init__(self, frac: float = 0.02, window: int = 8, season_slots: bool = True,
                 min_per_env: int = 64, val_cap: int = 500):
        self.frac, self.window = float(frac), int(window)
        self.season_slots = bool(season_slots)
        self.min_per_env, self.val_cap = int(min_per_env), int(val_cap)
        self.envs: dict = {}  # block -> dict(u, i, vu, vi, season)

    def select(self, u: np.ndarray, i: np.ndarray, priority: np.ndarray, rng: np.random.Generator):
        n = len(u)
        if n == 0:
            return np.zeros(0, np.int64)
        budget = min(n, max(self.min_per_env, int(np.ceil(self.frac * n))))
        w = np.clip(priority.astype(np.float64), 1e-6, None)
        keys = rng.random(n) ** (1.0 / w)  # A-Res
        return np.argsort(-keys)[:budget]

    def insert(self, block: int, season: int, u, i, vu, vi, rng: np.random.Generator):
        if len(vu) > self.val_cap:
            s = rng.choice(len(vu), self.val_cap, replace=False)
            vu, vi = vu[s], vi[s]
        self.envs[int(block)] = dict(u=np.asarray(u, np.int64), i=np.asarray(i, np.int64),
                                     vu=np.asarray(vu, np.int64), vi=np.asarray(vi, np.int64),
                                     season=int(season))
        return self._evict()

    def _evict(self):
        blocks = sorted(self.envs)
        keep = set(blocks[-self.window:])
        if self.season_slots:
            seen = set(self.envs[b]["season"] for b in keep)
            for b in reversed(blocks):
                s = self.envs[b]["season"]
                if b not in keep and s not in seen:
                    keep.add(b)
                    seen.add(s)
        ev_u, ev_i = [], []
        for b in blocks:
            if b not in keep:
                ev_u.append(self.envs[b]["u"])
                ev_i.append(self.envs[b]["i"])
                del self.envs[b]
        if ev_u:
            return np.concatenate(ev_u), np.concatenate(ev_i)
        return np.zeros(0, np.int64), np.zeros(0, np.int64)

    def env_ids(self):
        return sorted(self.envs)

    def size(self) -> int:
        return int(sum(len(e["u"]) for e in self.envs.values()))

    def all_pairs(self):
        if not self.envs:
            z = np.zeros(0, np.int64)
            return z, z, z
        us, is_, es = [], [], []
        for b, e in self.envs.items():
            us.append(e["u"])
            is_.append(e["i"])
            es.append(np.full(len(e["u"]), b, np.int64))
        return np.concatenate(us), np.concatenate(is_), np.concatenate(es)

    def sample(self, n: int, rng: np.random.Generator, mode: str = "pooled", now: int = 0,
               decay: float = 0.8):
        """pooled: uniform over all stored edges (v1; large old blocks dominate).
        env_recency: every environment gets weight decay**(now - block), edges uniform inside it,
        so the replay reminds of the *recent* past instead of the largest past."""
        u, i, e = self.all_pairs()
        if len(u) == 0 or n <= 0:
            return u[:0], i[:0], e[:0]
        if mode == "env_recency":
            blocks = np.asarray(sorted(self.envs), np.int64)
            w = np.power(float(decay), np.maximum(now - blocks, 0).astype(np.float64))
            w = w / w.sum()
            pe = np.searchsorted(blocks, e)  # env slot of every stored edge
            cnt = np.bincount(pe, minlength=len(blocks)).astype(np.float64)
            p = (w / np.maximum(cnt, 1.0))[pe]
            p = p / p.sum()
            s = rng.choice(len(u), size=n, replace=True, p=p)
            return u[s], i[s], e[s]
        s = rng.choice(len(u), size=n, replace=n > len(u))
        return u[s], i[s], e[s]

    def val_pairs(self):
        if not self.envs:
            z = np.zeros(0, np.int64)
            return z, z
        return (np.concatenate([e["vu"] for e in self.envs.values()]),
                np.concatenate([e["vi"] for e in self.envs.values()]))

    def state_dict(self):
        return {"envs": self.envs}

    def load_state_dict(self, d):
        self.envs = d["envs"]
