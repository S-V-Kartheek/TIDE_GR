"""Leak-free per-block features computed from the stream:

* environment covariates x_i^e (v1 Eq. 1): [log(1+n_i^e), trend, log(1+age_i^e), season one-hot]
  n_i^e = train-split count of item i in block e; age in blocks since first appearance;
* forecast x_hat^k from blocks k-1, k-2 (v1 Eq. 9);
* recent-interest index matrix: for every user, the last n items (train+val
  splits) strictly before block k.
"""
from __future__ import annotations

import numpy as np

from .data.stream import PART_TRAIN, PART_VAL, StreamData


class FeatureBank:
    def __init__(self, data: StreamData):
        self.data = data
        self._cnt: dict = {}
        self._recent: dict = {}
        self.p = 3 + data.n_seasons

    def counts(self, b: int) -> np.ndarray:
        if b < 0:
            return np.zeros(self.data.n_items, np.float32)
        if b not in self._cnt:
            self._cnt[b] = self.data.item_counts(b)
        return self._cnt[b]

    def _season(self, b: int) -> np.ndarray:
        oh = np.zeros(self.data.n_seasons, np.float32)
        oh[int(self.data.block_season[min(max(b, 0), self.data.n_blocks - 1)])] = 1.0
        return oh

    def _age(self, b: int) -> np.ndarray:
        return np.log1p(np.maximum(b - self.data.item_first_block, 0)).astype(np.float32)

    def x_env(self, e: int) -> np.ndarray:
        c = np.log1p(self.counts(e))
        c1 = np.log1p(self.counts(e - 1))
        x = np.zeros((self.data.n_items, self.p), np.float32)
        x[:, 0] = c
        x[:, 1] = c - c1
        x[:, 2] = self._age(e)
        x[:, 3:] = self._season(e)
        return x

    def x_forecast(self, k: int, eta: float) -> np.ndarray:
        c1 = np.log1p(self.counts(k - 1))
        c2 = np.log1p(self.counts(k - 2))
        c3 = np.log1p(self.counts(k - 3))
        t1, t2 = c1 - c2, c2 - c3
        x = np.zeros((self.data.n_items, self.p), np.float32)
        x[:, 0] = np.maximum(c1 + eta * t1, 0.0)
        x[:, 1] = t1 + eta * (t1 - t2)
        x[:, 2] = self._age(k)
        x[:, 3:] = self._season(k)
        return x

    def fit_eta(self, upto: int, grid=(0.0, 0.25, 0.5, 0.75, 1.0)) -> float:
        """Choose eta minimising the squared error of forecasting log-counts of
        past blocks (uses blocks < upto only)."""
        best, best_err = 0.5, np.inf
        blocks = [b for b in range(3, upto)]
        if not blocks:
            return best
        for eta in grid:
            err = 0.0
            for b in blocks:
                c1 = np.log1p(self.counts(b - 1))
                c2 = np.log1p(self.counts(b - 2))
                pred = np.maximum(c1 + eta * (c1 - c2), 0.0)
                err += float(((pred - np.log1p(self.counts(b))) ** 2).mean())
            if err < best_err:
                best, best_err = eta, err
        return best

    def recent(self, k: int, n: int) -> np.ndarray:
        """[n_users, n] int64 matrix of last-n items before block k (-1 = padding)."""
        key = (k, n)
        if key in self._recent:
            return self._recent[key]
        d = self.data
        end = int(d.block_ptr[max(k, 0)])
        sel = np.arange(end)
        sel = sel[np.isin(d.ev_part[:end], (PART_TRAIN, PART_VAL))]
        u, i, t = d.ev_user[sel], d.ev_item[sel], d.ev_ts[sel]
        R = np.full((d.n_users, n), -1, np.int64)
        if len(u):
            order = np.lexsort((t, u))
            u, i = u[order], i[order]
            cnt = np.bincount(u, minlength=d.n_users)
            ends = np.cumsum(cnt)
            rank_from_end = ends[u] - np.arange(len(u)) - 1
            keep = rank_from_end < n
            R[u[keep], n - 1 - rank_from_end[keep]] = i[keep]
        if len(self._recent) > 64:
            self._recent.pop(next(iter(self._recent)))
        self._recent[key] = R
        return R
