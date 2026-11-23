"""Aggregate finished runs into paper tables + significance tests.

  python -m tidegr.report --runs_root <root> --dataset kuairand [--reference tidegr]

Writes <root>/<dataset>/results/{summary.csv, table.md, significance.csv} and the paper
figures in results/figures/ (see tidegr.plots).
Significance: paired Wilcoxon signed-rank test on per-(user, block, seed)
NDCG@20 of the reference vs. each other run (matched seeds), Holm-corrected.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from .utils import load_json

MAIN = ["acc_recall@20", "acc_ndcg@20", "acc_ndcg@10", "rel_ndcg@20", "ts_ndcg@20", "ece", "nll", "brier",
        "aurc", "selective_err", "coverage", "aci_long_run_err", "bwt_recall@20",
        "final_retention_recall@20", "new_user_recall@20", "new_item_recall@20",
        "unbiased_acc_auc", "unbiased_acc_ndcg@10", "unbiased_ts_auc", "unbiased_ts_ndcg@10",
        "drift_auc", "drift_auc_lowactivity", "wall_hours"]


def collect(ds_dir: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(ds_dir, "*", "seed*", "summary.json")):
        s = load_json(f, {})
        run = os.path.basename(os.path.dirname(os.path.dirname(f)))
        if run == "tuning":
            continue
        s["run"] = run
        s["seed_dir"] = os.path.basename(os.path.dirname(f))
        rows.append(s)
    return pd.DataFrame(rows)


def table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in MAIN if c in df.columns]
    g = df.groupby("run")[cols]
    mean, std, n = g.mean(), g.std(), g.size()
    out = mean.copy()
    for c in cols:
        out[c] = [f"{m:.4f} ± {s:.4f}" if np.isfinite(s) else (f"{m:.4f}" if np.isfinite(m) else "")
                  for m, s in zip(mean[c], std[c].fillna(np.nan))]
    out.insert(0, "seeds", n)
    return out


def _per_user(run_dir: str) -> pd.DataFrame:
    parts = []
    for f in glob.glob(os.path.join(run_dir, "per_user", "block_*.npz")):
        z = np.load(f)
        b = int(os.path.basename(f)[6:-4])
        parts.append(pd.DataFrame({"user": z["users"], "block": b, "ndcg": z["ndcg"]}))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["user", "block", "ndcg"])


def significance(ds_dir: str, reference: str) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    ref_dirs = sorted(glob.glob(os.path.join(ds_dir, reference, "seed*")))
    rows = []
    for other in sorted(os.listdir(ds_dir)):
        od = os.path.join(ds_dir, other)
        if other in (reference, "tuning", "results", "data_cache") or not os.path.isdir(od):
            continue
        diffs = []
        for rd in ref_dirs:
            sd = os.path.join(od, os.path.basename(rd))
            if not os.path.isdir(sd):
                continue
            a, b = _per_user(rd), _per_user(sd)
            m = a.merge(b, on=["user", "block"], suffixes=("_ref", "_oth"))
            diffs.append(m["ndcg_ref"].to_numpy() - m["ndcg_oth"].to_numpy())
        if not diffs:
            continue
        d = np.concatenate(diffs)
        d = d[d != 0]
        p = float(wilcoxon(d).pvalue) if len(d) > 10 else float("nan")
        rows.append({"vs": other, "mean_diff_ndcg@20": float(np.mean(np.concatenate(diffs))),
                     "n_pairs": int(sum(len(x) for x in diffs)), "p_wilcoxon": p})
    res = pd.DataFrame(rows)
    if len(res):  # Holm correction
        order = np.argsort(res["p_wilcoxon"].fillna(1).to_numpy())
        m = len(res)
        adj = np.zeros(m)
        running = 0.0
        for rank, idx in enumerate(order):
            val = min(1.0, (m - rank) * float(res["p_wilcoxon"].fillna(1).iloc[idx]))
            running = max(running, val)
            adj[idx] = running
        res["p_holm"] = adj
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reference", default="tidegr")
    a = ap.parse_args(argv)
    ds = os.path.join(a.runs_root, a.dataset)
    out = os.path.join(ds, "results")
    os.makedirs(out, exist_ok=True)
    df = collect(ds)
    if df.empty:
        print("no finished runs yet")
        return
    df.to_csv(os.path.join(out, "summary_all_seeds.csv"), index=False)
    t = table(df)
    t.to_csv(os.path.join(out, "summary.csv"))
    with open(os.path.join(out, "table.md"), "w", encoding="utf-8") as f:
        try:
            f.write(t.to_markdown())
        except ImportError:  # tabulate not installed
            f.write(t.to_string())
    sig = significance(ds, a.reference)
    sig.to_csv(os.path.join(out, "significance.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.to_string())
    print(sig.to_string())
    try:
        from .plots import make_all
        make_all(a.runs_root, a.dataset, a.reference)
    except Exception as e:  # figures must never break the tables
        print(f"[plots] skipped: {e!r}")


if __name__ == "__main__":
    main()
