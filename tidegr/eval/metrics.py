"""Full-ranking top-K metrics (chunked, GPU) and calibration / selective metrics."""
from __future__ import annotations

import numpy as np
import torch


class CSR:
    """Per-row item lists aligned with a user array."""

    def __init__(self, indptr: np.ndarray, indices: np.ndarray):
        self.indptr, self.indices = indptr.astype(np.int64), indices.astype(np.int64)

    @staticmethod
    def build(rows_users: np.ndarray, u: np.ndarray, i: np.ndarray, n_users_total: int) -> "CSR":
        """Rows follow `rows_users`; entries are items i of pairs (u, i)."""
        pos = np.full(n_users_total, -1, np.int64)
        pos[rows_users] = np.arange(len(rows_users))
        r = pos[u]
        keep = r >= 0
        r, it = r[keep], i[keep]
        order = np.lexsort((it, r))
        r, it = r[order], it[order]
        if len(r):  # de-duplicate (row, item)
            d = np.ones(len(r), bool)
            d[1:] = (r[1:] != r[:-1]) | (it[1:] != it[:-1])
            r, it = r[d], it[d]
        indptr = np.zeros(len(rows_users) + 1, np.int64)
        np.add.at(indptr, r + 1, 1)
        return CSR(np.cumsum(indptr), it)

    def row(self, k: int) -> np.ndarray:
        return self.indices[self.indptr[k]: self.indptr[k + 1]]

    def lengths(self) -> np.ndarray:
        return np.diff(self.indptr)

    def slice_pairs(self, a: int, b: int):
        """(row offset within [a,b), item) pairs for rows a..b-1."""
        s, e = self.indptr[a], self.indptr[b]
        lens = np.diff(self.indptr[a: b + 1])
        rows = np.repeat(np.arange(b - a), lens)
        return rows, self.indices[s:e]


@torch.no_grad()
def rank_eval(score_fn, users: np.ndarray, targets: CSR, mask: CSR | None, valid_items: torch.Tensor,
              ks=(10, 20), chunk: int = 1024, device="cpu", return_top: int = 0,
              extra_score_fns: dict | None = None):
    """Evaluate full ranking.

    score_fn(users LongTensor[B]) -> FloatTensor[B, n_items] (on device).
    Returns dict of per-user arrays: recall@k, ndcg@k, hits@k (+ 'top' [n_users, return_top]).
    extra_score_fns: {name: fn} evaluated with the same masks (e.g. reliability, thompson).
    """
    fns = {"acc": score_fn}
    if extra_score_fns:
        fns.update(extra_score_fns)
    kmax = max(max(ks), return_top)
    n = len(users)
    n_items = valid_items.shape[0]
    out = {name: {f"{m}@{k}": np.zeros(n, np.float32) for k in ks for m in ("recall", "ndcg", "hits")}
           for name in fns}
    tops = {name: np.zeros((n, return_top), np.int64) for name in fns} if return_top else None
    tlen = targets.lengths().astype(np.float32)
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))
    idcg = {k: np.array([disc[: min(int(t), k)].sum() for t in range(0, k + 1)]) for k in ks}
    invalid = ~valid_items.to(device)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        ub = torch.as_tensor(users[a:b], device=device)
        tr, ti = targets.slice_pairs(a, b)
        T = torch.zeros((b - a, n_items), dtype=torch.bool, device=device)
        T[torch.as_tensor(tr, device=device), torch.as_tensor(ti, device=device)] = True
        if mask is not None:
            mr, mi = mask.slice_pairs(a, b)
            mr_t, mi_t = torch.as_tensor(mr, device=device), torch.as_tensor(mi, device=device)
        for name, fn in fns.items():
            S = fn(ub).float()
            S = S.masked_fill(invalid.unsqueeze(0), float("-inf"))
            if mask is not None and len(mr):
                S[mr_t, mi_t] = float("-inf")
            top = torch.topk(S, kmax, dim=1).indices
            hit = torch.gather(T, 1, top).float().cpu().numpy()
            tl = np.maximum(tlen[a:b], 1.0)
            for k in ks:
                h = hit[:, :k]
                hk = h.sum(1)
                out[name][f"hits@{k}"][a:b] = hk
                out[name][f"recall@{k}"][a:b] = hk / tl
                dcg = (h * disc[:k]).sum(1)
                ideal = idcg[k][np.minimum(tlen[a:b].astype(int), k)]
                out[name][f"ndcg@{k}"][a:b] = np.where(ideal > 0, dcg / np.maximum(ideal, 1e-12), 0.0)
            if return_top:
                tops[name][a:b] = top[:, :return_top].cpu().numpy()
    if return_top:
        for name in fns:
            out[name]["top"] = tops[name]
    return out


# --------------------------------------------------------------------------- calibration
def ece_score(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / len(p))


def nll_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(p) else float("nan")


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p - y) ** 2).mean()) if len(p) else float("nan")


def aurc(conf: np.ndarray, err: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    if len(conf) == 0:
        return float("nan")
    order = np.argsort(-conf, kind="stable")
    e = err[order].astype(np.float64)
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return float(risk.mean())


def auc_score(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
