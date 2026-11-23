"""Paper figures from finished runs, saved as PDF (vector, for LaTeX) + PNG (preview).

  python -m tidegr.plots --runs_root <root> --dataset kuairand [--reference tidegr]

Writes <root>/<dataset>/results/figures/ (on the shared Drive folder):
  stream_<metric>.pdf/png   metric per stream block, one line per method (mean over seeds, ±1 std band)
  bars_<metric>.pdf/png     overall metric per method (mean ± std over seeds); reference highlighted
  ablation_delta_ndcg.pdf/png   change of NDCG@20 of each ablation vs. the full reference model
Ablation runs are the ones whose name starts with "abl_"; everything else is a method.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from .report import collect
from .utils import load_json

# Fixed categorical order (validated palette, light mode). Colour follows the method, never its rank.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDER = ["tidegr", "lightgcn_ft", "xsimgcl_ft", "sasrec", "pda", "ewc", "kd_lite", "sqn",
         "lightgcn_retrain", "ensemble"]
LABEL = {"tidegr": "TIDE-GR", "lightgcn_ft": "LightGCN-FT", "xsimgcl_ft": "XSimGCL-FT", "sasrec": "SASRec",
         "pda": "PDA", "ewc": "EWC", "kd_lite": "KD-lite", "sqn": "SQN", "lightgcn_retrain": "LightGCN-retrain",
         "ensemble": "Ensemble"}
INK, INK2, GRID, NEUTRAL, NEG = "#0b0b0b", "#52514e", "#e4e3df", "#a9a8a2", "#e34948"
STREAM = {"acc_ndcg@20": "NDCG@20", "acc_recall@20": "Recall@20", "ece": "ECE (lower is better)",
          "coverage": "Coverage (selective)", "retention_mean_recall@20": "Retention Recall@20",
          "unbiased_acc_ndcg@10": "Unbiased NDCG@10 (random exposure)"}
BARS = {"acc_ndcg@20": "NDCG@20", "acc_recall@20": "Recall@20", "ece": "ECE (lower is better)",
        "aurc": "AURC (lower is better)", "bwt_recall@20": "Backward transfer (Recall@20)",
        "new_user_recall@20": "New-user Recall@20", "new_item_recall@20": "New-item Recall@20",
        "wall_hours": "Wall-clock hours"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "figure.facecolor": "white"})
    return plt


def style(run: str, i_extra: int = 0) -> dict:
    """Colour + dash per method; methods past the 8th colour repeat a hue with a dashed line + marker."""
    idx = ORDER.index(run) if run in ORDER else len(ORDER) + i_extra
    return {"color": PALETTE[idx % 8], "linestyle": "-" if idx < 8 else "--",
            "marker": None if idx < 8 else "o", "label": LABEL.get(run, run)}


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return os.path.join(out, f"{name}.png")


def _blocks(ds_dir: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(ds_dir, "*", "seed*", "metrics.json")):
        run = os.path.basename(os.path.dirname(os.path.dirname(f)))
        if run == "tuning":
            continue
        for r in load_json(f, []) or []:
            if "block" in r:
                rows.append({"run": run, "seed": os.path.basename(os.path.dirname(f)),
                             **{k: v for k, v in r.items() if isinstance(v, (int, float))}})
    return pd.DataFrame(rows)


def _methods(runs) -> list:
    ms = [r for r in runs if not str(r).startswith("abl_")]
    return sorted(ms, key=lambda r: (ORDER.index(r) if r in ORDER else 99, r))


def stream_figs(blk: pd.DataFrame, out: str, plt) -> list:
    paths = []
    if blk.empty:
        return paths
    runs = _methods(blk["run"].unique())
    extras = [r for r in runs if r not in ORDER]
    for metric, label in STREAM.items():
        if metric not in blk.columns or blk[metric].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        for run in runs:
            d = blk[blk["run"] == run].groupby("block")[metric].agg(["mean", "std"]).dropna(subset=["mean"])
            if d.empty:
                continue
            st = style(run, extras.index(run) if run in extras else 0)
            lw = 2.0 if run == "tidegr" else 1.4
            ax.plot(d.index, d["mean"], lw=lw, markersize=4, zorder=3 if run == "tidegr" else 2, **st)
            if d["std"].notna().any():
                ax.fill_between(d.index, d["mean"] - d["std"].fillna(0), d["mean"] + d["std"].fillna(0),
                                color=st["color"], alpha=0.12, lw=0)
        ax.set_xlabel("Stream block")
        ax.set_ylabel(label)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        paths.append(_save(fig, out, f"stream_{metric.replace('@', '')}"))
    return paths


def bar_figs(df: pd.DataFrame, out: str, plt, reference: str) -> list:
    paths = []
    if df.empty:
        return paths
    runs = _methods(df["run"].unique())
    for metric, label in BARS.items():
        if metric not in df.columns:
            continue
        g = df[df["run"].isin(runs)].groupby("run")[metric].agg(["mean", "std", "size"]).dropna(subset=["mean"])
        if g.empty:
            continue
        g = g.reindex([r for r in runs if r in g.index])
        fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(g) + 0.9))
        y = np.arange(len(g))[::-1]
        colors = [PALETTE[0] if r == reference else NEUTRAL for r in g.index]
        ax.barh(y, g["mean"], xerr=g["std"].fillna(0), color=colors, height=0.62,
                error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
        ax.set_yticks(y, [LABEL.get(r, r) for r in g.index])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(label)
        span = float(np.nanmax(np.abs(g["mean"]))) or 1.0
        for yi, m, sd in zip(y, g["mean"], g["std"].fillna(0)):  # label sits past the error bar
            ax.text(m + (sd + 0.015 * span) * (1 if m >= 0 else -1), yi, f"{m:.4f}", va="center",
                    ha="left" if m >= 0 else "right", fontsize=7, color=INK2)
        ax.margins(x=0.12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        paths.append(_save(fig, out, f"bars_{metric.replace('@', '')}"))
    return paths


def ablation_fig(df: pd.DataFrame, out: str, plt, reference: str, metric: str = "acc_ndcg@20") -> list:
    if df.empty or metric not in df.columns:
        return []
    ref = df[df["run"] == reference]
    abl = df[df["run"].astype(str).str.startswith("abl_")]
    if ref.empty or abl.empty:
        return []
    # compare on the seeds both have (ablations usually run seed 0 only)
    rows = []
    for run, d in abl.groupby("run"):
        r = ref[ref["seed_dir"].isin(d["seed_dir"])]
        if len(r):
            rows.append((run, float(d[metric].mean() - r[metric].mean())))
    if not rows:
        return []
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(rows) + 0.9))
    y = np.arange(len(rows))
    vals = [v for _, v in rows]
    ax.barh(y, vals, color=[NEG if v < 0 else PALETTE[0] for v in vals], height=0.62)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [r[4:].replace("_", " ") for r, _ in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Δ NDCG@20 vs. full {LABEL.get(reference, reference)}")
    span = max(abs(v) for v in vals) or 1.0
    for yi, v in zip(y, vals):
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), yi, f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7, color=INK2)
    ax.margins(x=0.18)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    return [_save(fig, out, "ablation_delta_ndcg20")]


def make_all(runs_root: str, dataset: str, reference: str = "tidegr") -> list:
    ds_dir = os.path.join(runs_root, dataset)
    out = os.path.join(ds_dir, "results", "figures")
    os.makedirs(out, exist_ok=True)
    plt = _plt()
    df = collect(ds_dir)
    paths = stream_figs(_blocks(ds_dir), out, plt) + bar_figs(df, out, plt, reference) \
        + ablation_fig(df, out, plt, reference)
    print(f"[plots] {len(paths)} figures -> {out}")
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reference", default="tidegr")
    a = ap.parse_args(argv)
    make_all(a.runs_root, a.dataset, a.reference)


if __name__ == "__main__":
    main()
