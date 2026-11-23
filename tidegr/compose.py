"""Build the final config of one run:
base.yaml  <- data yaml  <- method yaml  <- data.method_overrides  <- plan overrides
<- tuned parameters  -> resolve 'auto' values from the dataset defaults."""
from __future__ import annotations

import os

from .config import Cfg, _load_file, _wrap, deep_merge, parse_overrides, set_path, to_plain
from .utils import load_json

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def _p(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(CONFIG_DIR, path)


def _resolve_auto(d: dict):
    auto = d.get("auto", {}) or {}
    m = d.get("method", {})
    pairs = [("recent", "n", "recent_n"), ("edge", "tau_rho", "tau_rho"), ("memory", "window", "memory_window")]
    defaults = {"recent_n": 20, "tau_rho": 4.0, "memory_window": 8}
    for sec, key, ak in pairs:
        if isinstance(m.get(sec), dict) and m[sec].get(key) == "auto":
            m[sec][key] = auto.get(ak, defaults[ak])
    if m.get("max_len") == "auto":
        m["max_len"] = 50


def compose(data_yaml: str, method_yaml: str, overrides: dict | list | None = None,
            tuned: dict | None = None, extra: dict | None = None) -> Cfg:
    d = _load_file(_p("base.yaml"))
    d = deep_merge(d, _load_file(_p(data_yaml)))
    meth = _load_file(_p(method_yaml))
    d = deep_merge(d, meth)
    if d.get("method_overrides"):
        d["method"] = deep_merge(d["method"], d["method_overrides"])
    if overrides:
        ov = overrides if isinstance(overrides, dict) else parse_overrides(overrides)
        flat = {}
        for k, v in ov.items():
            if "." in k:
                set_path(flat, k, v)
            else:
                flat = deep_merge(flat, {k: v})
        d = deep_merge(d, flat)
    if tuned:
        t = {}
        for k, v in tuned.items():
            set_path(t, k, v)
        d = deep_merge(d, t)
    if extra:
        d = deep_merge(d, extra)
    _resolve_auto(d)
    return _wrap(d)


def tuned_params(runs_root: str, dataset: str, name: str) -> dict | None:
    r = load_json(os.path.join(runs_root, dataset, "tuning", f"{name}.json"))
    return None if r is None or not r.get("finished") else r.get("best_params", {})
