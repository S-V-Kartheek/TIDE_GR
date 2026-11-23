"""Seeding, RNG state, atomic checkpoint I/O, logging and time budget."""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch


# --------------------------------------------------------------------------- device / seed
def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state(np_rng: np.random.Generator | None = None) -> dict:
    st = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    if np_rng is not None:
        st["np_rng"] = np_rng.bit_generator.state
    return st


def set_rng_state(st: dict, np_rng: np.random.Generator | None = None):
    random.setstate(st["python"])
    np.random.set_state(st["numpy_global"])
    torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu") else st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])
        except Exception:  # different number of GPUs on the new machine
            pass
    if np_rng is not None and "np_rng" in st:
        np_rng.bit_generator.state = st["np_rng"]


# --------------------------------------------------------------------------- atomic I/O
def atomic_torch_save(obj: Any, path: str, keep_prev: bool = True):
    """Write to tmp, fsync, keep previous copy, then rename (crash-safe)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    if keep_prev and os.path.exists(path):
        prev = path + ".prev"
        try:
            shutil.copyfile(path, prev)
        except OSError:
            pass
    os.replace(tmp, path)


def safe_torch_load(path: str, map_location="cpu"):
    """Load a checkpoint; fall back to the .prev copy if the main one is corrupt."""
    for p in (path, path + ".prev"):
        if os.path.exists(p):
            try:
                return torch.load(p, map_location=map_location, weights_only=False)
            except Exception as e:  # corrupted / partially written
                log(f"[ckpt] failed to load {p}: {e!r}")
    return None


def atomic_json_save(obj: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def append_jsonl(path: str, rec: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=_json_default) + "\n")
        f.flush()


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a torn last line after a crash
    return out


# --------------------------------------------------------------------------- logging / time
_LOG_FILE: str | None = None


def set_log_file(path: str | None):
    global _LOG_FILE
    _LOG_FILE = path
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE:
        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


class TimeBudget:
    """Stops work cleanly before a Colab session limit."""

    def __init__(self, max_hours: float | None):
        self.start = time.time()
        self.max_s = None if not max_hours else float(max_hours) * 3600.0

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return float("inf") if self.max_s is None else self.max_s - self.elapsed()

    def ok(self, need_s: float = 0.0) -> bool:
        return self.remaining() > need_s


class TimeLimitReached(Exception):
    pass


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def stderr(msg: str):
    print(msg, file=sys.stderr, flush=True)
