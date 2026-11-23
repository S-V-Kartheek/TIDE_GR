"""Small synthetic stream with popularity drift, seasonal items and preference
drift. Used for local testing of the whole pipeline (CPU, < 2 minutes)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stream import build_stream


def load(cfg, raw_dir: str, seed: int = 0):
    d = cfg.data
    rng = np.random.default_rng(seed)
    n_u, n_i, k = int(d.get("n_users", 800)), int(d.get("n_items", 500)), 8
    n_b, n_w = int(d.get("n_blocks", 10)), int(d.get("n_warmup", 5))
    per_user = float(d.get("per_user_per_block", 4.0))
    U = rng.normal(size=(n_u, k))
    V = rng.normal(size=(n_i, k))
    item_birth = rng.integers(0, n_b, size=n_i)
    item_birth[: n_i // 2] = 0  # half of the catalogue exists from the start
    user_birth = rng.integers(0, n_b, size=n_u)
    user_birth[: n_u // 2] = 0
    season_items = rng.random(n_i) < 0.1
    drifters = rng.random(n_u) < 0.15
    drift_at = n_w + 2
    U2 = rng.normal(size=(n_u, k))
    rows = []
    for b in range(n_b):
        pop = np.exp(rng.normal(0, 0.5, n_i))  # block-specific popularity
        if b % 4 == 3:
            pop[season_items] *= 5.0  # seasonal spike
        exists = item_birth <= b
        for u in np.where(user_birth <= b)[0]:
            n = rng.poisson(per_user)
            if n == 0:
                continue
            pu = U2[u] if (drifters[u] and b >= drift_at) else U[u]
            logit = V @ pu / np.sqrt(k) * 2.0 + np.log(pop)
            logit[~exists] = -1e9
            p = np.exp(logit - logit.max())
            p /= p.sum()
            items = rng.choice(n_i, size=min(n, int(exists.sum())), replace=False, p=p)
            base = b * 86400 * 7
            for it in items:
                rows.append((u, it, base + int(rng.integers(0, 86400 * 7)), b, 1.0))
    df = pd.DataFrame(rows, columns=["user", "item", "ts", "block", "reward"])
    feat = {"ids": np.arange(n_i), "X": (V + rng.normal(0, 0.3, V.shape)).astype(np.float32)}
    # random exposure in stream blocks
    rr = []
    for b in range(n_w, n_b):
        for u in rng.choice(n_u, size=n_u // 4, replace=False):
            for it in rng.choice(n_i, size=10, replace=False):
                pu = U2[u] if (drifters[u] and b >= drift_at) else U[u]
                pr = 1 / (1 + np.exp(-(V[it] @ pu / np.sqrt(k) * 2.0 - 1.0)))
                rr.append((u, it, b * 86400 * 7, b, int(rng.random() < pr)))
    rnd = pd.DataFrame(rr, columns=["user", "item", "ts", "block", "label"])
    return build_stream(
        df, name="synthetic", block_names=[f"b{b}" for b in range(n_b)],
        block_season=[b % 4 for b in range(n_b)], n_seasons=4, n_warmup=n_w,
        val_frac=0.1, ret_frac=0.05, seed=seed, item_feat_raw=feat, random_df=rnd,
        swap_frac=float(d.get("swap_frac", 0.0)), swap_stream_offset=int(d.get("swap_stream_offset", 2)),
        aci_round_seconds=86400, meta={"drifters": int(drifters.sum())})
