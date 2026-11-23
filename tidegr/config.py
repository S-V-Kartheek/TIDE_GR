"""YAML configuration with inheritance and command-line overrides.

A config file may contain ``inherit: [other.yaml, ...]``; inherited files are
merged first (deep merge), then the file itself, then CLI overrides of the
form ``key.sub=value`` (values parsed as YAML).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Iterable

import yaml


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return v

    def __setattr__(self, k, v):
        self[k] = v

    def get_path(self, path: str, default=None):
        cur: Any = self
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def to_plain(x):
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_plain(v) for v in x]
    return x


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    base: dict = {}
    for inh in d.pop("inherit", []) or []:
        p = inh if os.path.isabs(inh) else os.path.join(os.path.dirname(path), inh)
        base = deep_merge(base, _load_file(p))
    return deep_merge(base, d)


def set_path(d: dict, path: str, value):
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def parse_overrides(items: Iterable[str]) -> dict:
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"override must be key=value, got {it!r}")
        k, v = it.split("=", 1)
        set_path(out, k.strip(), yaml.safe_load(v))
    return out


def load_config(path: str, overrides: Iterable[str] | dict | None = None) -> Cfg:
    d = _load_file(path)
    if overrides:
        ov = overrides if isinstance(overrides, dict) else parse_overrides(overrides)
        d = deep_merge(d, ov)
    return _wrap(d)


def merge_cfg(cfg: Cfg, extra: dict) -> Cfg:
    return _wrap(deep_merge(to_plain(cfg), to_plain(extra)))


def cfg_hash(cfg) -> str:
    s = json.dumps(to_plain(cfg), sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:10]
