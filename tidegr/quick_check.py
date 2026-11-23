"""Quick check before the full run (about 15-30 minutes on a T4).

  python -m tidegr.quick_check --data data/kuairand.yaml --runs_root /content/drive/MyDrive/TIDE_GR_runs

1. builds / loads the dataset and prints its statistics;
2. runs TIDE-GR, LightGCN-FT and SASRec on a short stream (capped warm-up + 2 blocks);
3. checks: finite metrics, better than a popularity ranking, resume from checkpoint works;
4. times every phase and extrapolates the full plan to GPU-hours (printed + quick_check.json).
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import shutil
import time

import numpy as np
import torch

from .compose import _load_file, _p, compose
from .data.registry import load_data
from .data.stream import PART_TRAIN
from .runner import Runner
from .utils import atomic_json_save, get_device, log


def popularity_recall(data, k_block, K=20):
    """Recall@K of 'most popular items in the previous block' on block k (reference floor)."""
    cnt = data.item_counts(k_block - 1)
    top = np.argsort(-cnt)[:K]
    u, i, _, _ = data.pairs(k_block, (0, 1, 2))
    rec = []
    for uu in np.unique(u)[:20000]:
        t = i[u == uu]
        rec.append(np.isin(t, top).sum() / len(t))
    return float(np.mean(rec)) if rec else 0.0


def run_short(data, data_yaml, method_yaml, root, device, warm_cap, n_blocks, name):
    view = dataclasses.replace(data, stream_blocks=data.stream_blocks[:n_blocks])
    cfg = compose(data_yaml, method_yaml, extra={"method": {"epochs": {"warmup_max": warm_cap}}})
    run_dir = os.path.join(root, name)
    shutil.rmtree(run_dir, ignore_errors=True)
    t0 = time.time()
    r = Runner(cfg, view, run_dir, seed=0, device=device)
    summ = r.run()
    total = time.time() - t0
    warm = next((x for x in r.records if x.get("phase") == "warmup"), {})
    blocks = [x for x in r.records if "block" in x]
    out = {
        "total_s": total,
        "warmup_s": warm.get("time_s"),
        "warmup_epochs": warm.get("warmup_epochs"),
        "warmup_epoch_s": warm.get("warmup_epoch_s"),
        "block_s": float(np.mean([b["block_s"] for b in blocks])) if blocks else None,
        "acc_recall@20": summ.get("acc_recall@20"),
        "acc_ndcg@20": summ.get("acc_ndcg@20"),
        "ece": summ.get("ece"),
        "finite": all(np.isfinite(v) for v in summ.values() if isinstance(v, float)),
        "cfg": cfg,
        "view": view,
        "run_dir": run_dir,
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="e.g. data/kuairand.yaml")
    ap.add_argument("--plan", default=None, help="plan to extrapolate (default: plans/<dataset>.yaml)")
    ap.add_argument("--runs_root", default=None)
    ap.add_argument("--warmup_epochs", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--methods", default="tidegr,lightgcn_ft,sasrec")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    device = get_device(a.device)
    probe = compose(a.data, "methods/tidegr.yaml")
    root = a.runs_root or probe.get("runs_root", "./runs")
    t0 = time.time()
    data = load_data(probe, root)
    t_data = time.time() - t0
    qc = os.path.join(root, data.name, "quick_check")
    log(f"[qc] device={device} data built/loaded in {t_data:.0f}s: {data.summary()}")
    if device.type != "cuda":
        log("[qc] WARNING: no GPU visible. In Colab: Runtime > Change runtime type > T4 GPU.")

    res = {}
    for m in [x.strip() for x in a.methods.split(",") if x.strip()]:
        log(f"[qc] ---- {m}: warm-up capped at {a.warmup_epochs} epochs + {a.blocks} stream blocks")
        res[m] = run_short(data, a.data, f"methods/{m}.yaml", qc, device, a.warmup_epochs, a.blocks, m)
        log(f"[qc] {m}: recall@20={res[m]['acc_recall@20']:.4f} ndcg@20={res[m]['acc_ndcg@20']:.4f} "
            f"warm-up epoch={res[m]['warmup_epoch_s']:.1f}s block={res[m]['block_s']:.1f}s")

    # --- resume self-test: continue the TIDE-GR short run by one more block from its checkpoint
    resume_ok = None
    if "tidegr" in res and len(data.stream_blocks) > a.blocks:
        r0 = res["tidegr"]
        view3 = dataclasses.replace(data, stream_blocks=data.stream_blocks[: a.blocks + 1])
        rr = Runner(r0["cfg"], view3, r0["run_dir"], seed=0, device=device)
        rr.load()
        resumed_at = rr.next_idx
        rr.phase = "stream"  # the short run was marked done; continue the stream
        rr.run(resume=False)
        resume_ok = resumed_at == a.blocks and rr.next_idx == a.blocks + 1
        log(f"[qc] resume test: restarted at stream block index {resumed_at} -> "
            f"{'OK' if resume_ok else 'FAILED'}")

    pop = popularity_recall(data, data.stream_blocks[0])
    checks = {
        "metrics_finite": all(r["finite"] for r in res.values()),
        "beats_popularity": {m: (r["acc_recall@20"] or 0) > pop for m, r in res.items()},
        "popularity_recall@20": pop,
        "resume_ok": resume_ok,
    }

    # --- extrapolate to the full plan
    n_stream = len(data.stream_blocks)
    plan_path = a.plan or f"plans/{data.name}.yaml"
    est = {}

    def full_hours(m, warm_epochs_full):
        r = res[m]
        ep = r["warmup_epoch_s"] or 0.0
        return (ep * warm_epochs_full + (r["block_s"] or 0.0) * n_stream) / 3600

    tid_cfg = compose(a.data, "methods/tidegr.yaml")
    wmax = int(tid_cfg.method.epochs.warmup_max)
    typical_epochs = min(wmax, 80)  # early stopping usually ends well before the cap
    per_method = {}
    for m in res:
        per_method[m] = {"typical_h": full_hours(m, typical_epochs), "worst_h": full_hours(m, wmax)}
    tid = per_method.get("tidegr", {"typical_h": 1.0, "worst_h": 1.5})
    ft = per_method.get("lightgcn_ft", {"typical_h": 0.6 * tid["typical_h"], "worst_h": 0.6 * tid["worst_h"]})
    sas = per_method.get("sasrec", {"typical_h": ft["typical_h"] * 1.5, "worst_h": ft["worst_h"] * 1.5})
    factor = {
        "tidegr": tid, "lightgcn_ft": ft, "xsimgcl_ft": {k: 1.3 * v for k, v in ft.items()},
        "pda": ft, "ewc": {k: 1.2 * v for k, v in ft.items()}, "kd_lite": ft, "sasrec": sas,
        "sqn": {k: 1.4 * v for k, v in sas.items()}, "ensemble": {k: 5 * v for k, v in ft.items()},
        "lightgcn_retrain": {k: (res.get("lightgcn_ft", {}).get("warmup_epoch_s") or 1.0) * typical_epochs
                             * n_stream / 3600 * (1.0 if k == "typical_h" else 1.5) for k in ("typical_h", "worst_h")},
    }
    try:
        plan = _load_file(_p(plan_path))
        rows = []
        tuned_done = set()
        for run in plan["runs"]:
            f = factor.get(run["name"], tid if run["name"].startswith(("abl_", "h3_")) else ft)
            n = len(run.get("seeds", [0]))
            rows.append((run["name"], n, n * f["typical_h"], n * f["worst_h"]))
            if run.get("tune") and run["name"] not in tuned_done:
                tuned_done.add(run["name"])
                grid = compose(a.data, run["method"]).get("tune", []) or []
                trials = 1 + sum(max(0, len(v) - 1) for _, v in grid)
                # a trial = capped warm-up (40 epochs max) on the earlier history + 3 history blocks
                g = factor.get(run["name"], ft)
                t_trial = 0.35 * g["typical_h"]
                rows.append((f"tune:{run['name']}", trials, trials * t_trial, trials * t_trial * 1.5))
        est = {"rows": rows, "total_typical_h": sum(r[2] for r in rows),
               "total_worst_h": sum(r[3] for r in rows)}
    except Exception as e:
        log(f"[qc] could not read plan {plan_path}: {e!r}")

    out = {"device": str(device), "data": data.summary(), "data_build_s": t_data,
           "short_runs": {m: {k: v for k, v in r.items() if k not in ("cfg", "view")} for m, r in res.items()},
           "per_method_full_run_h": per_method, "checks": checks, "plan_estimate": est}
    atomic_json_save(out, os.path.join(qc, "quick_check.json"))
    print("\n================ QUICK CHECK ================")
    print(f"device: {device}   dataset: {data.summary()}")
    for m, r in res.items():
        print(f"{m:14s} recall@20={r['acc_recall@20']:.4f} ndcg@20={r['acc_ndcg@20']:.4f} "
              f"warm-up epoch={r['warmup_epoch_s']:.1f}s  block={r['block_s']:.1f}s  "
              f"full run ~{per_method[m]['typical_h']:.2f}h (worst {per_method[m]['worst_h']:.2f}h)")
    print(f"popularity recall@20 (floor) = {pop:.4f}")
    print(f"checks: {checks}")
    if est:
        print("\nplan estimate (GPU-hours):")
        for name, n, t, w in est["rows"]:
            print(f"  {name:24s} x{n:<3d} typical {t:6.2f}h  worst {w:6.2f}h")
        print(f"  TOTAL typical {est['total_typical_h']:.1f}h, worst {est['total_worst_h']:.1f}h")
    print(f"\nsaved: {os.path.join(qc, 'quick_check.json')}")


if __name__ == "__main__":
    main()
