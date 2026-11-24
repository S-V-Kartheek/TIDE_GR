"""History-only hyper-parameter tuning (equal procedure for every method).

The last ``tune_blocks`` history blocks are used as a prequential mini-stream
(warm-up on the earlier history blocks, then test-then-train on these blocks);
the objective is the mean next-block NDCG@20. No stream data is touched.
Coordinate search over the method's ``tune`` grid; every trial is resumable.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import numpy as np

from .compose import compose
from .runner import Runner
from .utils import atomic_json_save, load_json, log, TimeLimitReached


def history_view(data, tune_blocks: int):
    W = list(data.warmup_blocks)
    tb = max(1, min(int(tune_blocks), len(W) - 2))
    return dataclasses.replace(data, warmup_blocks=W[:-tb], stream_blocks=W[-tb:],
                               rnd_user=None, rnd_item=None, rnd_label=None, rnd_block=None,
                               swapped_users=None, swap_block=None)


def _key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def tune_method(data, data_yaml, method_yaml, name, runs_root, device, budget, overrides=None,
                trial_epochs_cap: int | None = 40, extra=None):
    out_dir = os.path.join(runs_root, data.name, "tuning")
    res_path = os.path.join(out_dir, f"{name}.json")
    res = load_json(res_path, {"trials": {}, "best_params": {}, "best_score": None, "finished": False})
    if res.get("finished"):
        return res["best_params"]
    base_cfg = compose(data_yaml, method_yaml, overrides, extra=extra)
    grid = base_cfg.get("tune", []) or []
    view = history_view(data, int(base_cfg.get("tune_blocks", 4)))
    log(f"[tune] {name}: {len(grid)} parameters, mini-stream blocks={view.stream_blocks}")

    def trial(params: dict) -> float:
        # parameters equal to the config default are dropped, so the same configuration is never trained twice
        params = {p: v for p, v in params.items() if base_cfg.get_path(p) != v}
        k = _key(params)
        if k in res["trials"]:
            return res["trials"][k]
        ex = dict(extra or {})
        if trial_epochs_cap:
            from .config import deep_merge
            ex = deep_merge(ex, {"method": {"epochs": {"warmup_max": min(int(base_cfg.method.get(
                "epochs", {}).get("warmup_max", 200)), trial_epochs_cap)}}})
        cfg = compose(data_yaml, method_yaml, overrides, tuned=params, extra=ex)
        tid = hashlib.md5(k.encode()).hexdigest()[:10]
        run_dir = os.path.join(out_dir, name, f"trial_{tid}")
        r = Runner(cfg, view, run_dir, seed=0, device=device, max_hours=budget.remaining() / 3600)
        summ = r.run()
        score = float(summ.get("acc_ndcg@20", 0.0))
        res["trials"][k] = score
        atomic_json_save(res, res_path)
        log(f"[tune] {name} {params} -> ndcg@20={score:.5f}")
        return score

    best = dict(res.get("best_params", {}))
    best_score = trial(best)
    for path, values in grid:
        for v in values:
            cand = dict(best)
            cand[path] = v
            s = trial(cand)
            if s > best_score + 1e-5:
                best, best_score = cand, s
        res["best_params"], res["best_score"] = best, best_score
        atomic_json_save(res, res_path)
    res["finished"] = True
    atomic_json_save(res, res_path)
    log(f"[tune] {name} best {best} ndcg@20={best_score:.5f}")
    return best
