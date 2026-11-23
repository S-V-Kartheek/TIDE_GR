"""Amazon Reviews 2023, Video_Games, official 5-core ("rating_only") loader.

* Every review is a positive interaction.
* Blocks: one block for everything before ``history_start_year`` (default 2015),
  then one block per calendar quarter.
* Warm-up = blocks before ``stream_start_year`` (default 2018).
* Item side information: frozen text embedding of title + features +
  description + categories (all-MiniLM-L6-v2), cached on disk.
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd

from ..utils import log
from .download import download
from .stream import build_stream

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"


def _urls(category: str):
    return (f"{BASE}/benchmark/5core/rating_only/{category}.csv.gz",
            f"{BASE}/raw/meta_categories/meta_{category}.jsonl.gz")


def _item_text(meta_path: str, keep: set) -> pd.DataFrame:
    ids, texts = [], []
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = d.get("parent_asin")
            if pid not in keep:
                continue
            parts = [str(d.get("title") or "")]
            for k in ("features", "description", "categories"):
                v = d.get(k) or []
                if isinstance(v, list):
                    parts.append(" ".join(str(x) for x in v))
            if d.get("store"):
                parts.append(f"store: {d['store']}")
            texts.append(" | ".join(p for p in parts if p)[:1000])
            ids.append(pid)
    return pd.DataFrame({"item": ids, "text": texts})


def load(cfg, raw_dir: str, seed: int = 0):
    dcfg = cfg.data
    cat = dcfg.get("category", "Video_Games")
    r_url, m_url = _urls(cat)
    r_path = download(r_url, os.path.join(raw_dir, f"amazon_{cat}_5core.csv.gz"))
    log("[amazon] reading ratings")
    df = pd.read_csv(r_path)
    df = df.rename(columns={"user_id": "user", "parent_asin": "item"})
    ts = df["timestamp"].to_numpy(np.int64)
    if ts.max() > 10 ** 11:  # milliseconds
        ts = ts // 1000
    df["ts"] = ts
    df["reward"] = 1.0

    hs = int(dcfg.get("history_start_year", 2015))
    ss = int(dcfg.get("stream_start_year", 2018))
    dt = pd.to_datetime(df["ts"], unit="s", utc=True)
    q = (dt.dt.year - hs) * 4 + (dt.dt.quarter - 1)  # quarter index since history start
    df["block"] = np.where(dt.dt.year < hs, 0, q.to_numpy() + 1).astype(np.int64)
    n_blocks = int(df["block"].max()) + 1
    names = [f"<{hs}"] + [f"{hs + (b - 1) // 4}Q{(b - 1) % 4 + 1}" for b in range(1, n_blocks)]
    seasons = [0] + [(b - 1) % 4 for b in range(1, n_blocks)]
    n_warmup = 1 + (ss - hs) * 4
    # drop partial trailing blocks (the 5-core crawl ends mid-2023): < 25% of the median stream block
    counts = np.bincount(df["block"], minlength=n_blocks)
    med = float(np.median(counts[n_warmup:])) if n_blocks > n_warmup else 0.0
    while n_blocks > n_warmup + 1 and counts[n_blocks - 1] < 0.25 * med:
        df = df[df.block != n_blocks - 1]
        n_blocks -= 1
    names, seasons = names[:n_blocks], seasons[:n_blocks]

    feat = None
    if dcfg.get("use_text", True):
        cache = os.path.join(raw_dir, f"amazon_{cat}_text_emb.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            feat = {"ids": z["ids"], "X": z["X"]}
        else:
            m_path = download(m_url, os.path.join(raw_dir, f"meta_{cat}.jsonl.gz"))
            log("[amazon] parsing item metadata")
            meta = _item_text(m_path, set(df["item"].unique()))
            from .text import embed_texts
            X = embed_texts(meta["text"].tolist(), dcfg.get("text_model",
                            "sentence-transformers/all-MiniLM-L6-v2"), "cuda")
            np.savez(cache, ids=meta["item"].to_numpy(), X=X)
            feat = {"ids": meta["item"].to_numpy(), "X": X}

    return build_stream(
        df[["user", "item", "ts", "block", "reward"]],
        name=str(dcfg.get("name", f"amazon_{cat}")),
        block_names=names,
        block_season=seasons,
        n_seasons=4,
        n_warmup=n_warmup,
        val_frac=float(dcfg.get("val_frac", 0.1)),
        ret_frac=float(dcfg.get("ret_frac", 0.05)),
        seed=seed,
        item_feat_raw=feat,
        swap_frac=float(dcfg.get("swap_frac", 0.0)),
        swap_stream_offset=int(dcfg.get("swap_stream_offset", 3)),
        aci_round_seconds=int(dcfg.get("aci_round_seconds", 7 * 86400)),
        meta={"category": cat},
    )
