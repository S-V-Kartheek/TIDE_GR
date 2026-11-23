"""Dataset dispatcher with an on-disk cache (shared Drive folder => built once)."""
from __future__ import annotations

import os

import torch

from ..config import cfg_hash, to_plain
from ..utils import atomic_torch_save, log, safe_torch_load
from .stream import StreamData


def load_data(cfg, data_root: str) -> StreamData:
    name = cfg.data.name
    raw_dir = os.path.join(data_root, "raw")
    cache_dir = os.path.join(data_root, name, "data_cache")
    h = cfg_hash({"data": to_plain(cfg.data), "seed": int(cfg.get("data_seed", 0))})
    cache = os.path.join(cache_dir, f"stream_{h}.pt")
    obj = safe_torch_load(cache)
    if obj is not None:
        data = StreamData.from_dict(obj)
        log(f"[data] loaded cached stream {cache}")
        return data
    seed = int(cfg.get("data_seed", 0))
    if name == "kuairand":
        from . import kuairand as mod
    elif name.startswith("amazon"):
        from . import amazon as mod
    elif name == "synthetic":
        from . import synthetic as mod
    else:
        raise ValueError(f"unknown dataset {name}")
    data = mod.load(cfg, raw_dir, seed=seed)
    atomic_torch_save(data.to_dict(), cache, keep_prev=False)
    log(f"[data] built and cached {cache}: {data.summary()}")
    return data
