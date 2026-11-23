"""Mirror every finished result into one Google Sheet (for writing the paper).

In a Colab notebook cell (not via `!python`, the Google login lives in the notebook kernel):

    from tidegr.sheets import sync
    sync(RUNS_ROOT, "kuairand", SHEET)          # SHEET = sheet URL, key or name

One sheet holds all datasets; each dataset gets its own tabs, rewritten from the files on
Drive on every call (so it is always identical to Drive, safe to call any number of times):

    <ds>_table         mean ± std over seeds per run (the paper table)
    <ds>_runs          one row per run/seed, every summary metric
    <ds>_blocks        one row per run/seed/stream block (curves over time)
    <ds>_significance  Wilcoxon + Holm vs. the reference run
    <ds>_status        state of every run of the plan
    log                one line per sync (time, dataset, number of finished runs)

Login: the Colab Google login (`auth.authenticate_user()`), or a service-account JSON key
(`sa_key=` path, the sheet must be shared with the service account's e-mail).
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

import pandas as pd

from .report import collect, significance, table
from .utils import load_json

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]


def client(sa_key: str | None = None):
    import gspread

    if sa_key:
        return gspread.service_account(filename=sa_key, scopes=SCOPES)
    try:
        from google.colab import auth
        auth.authenticate_user()
    except ImportError:  # not on Colab: fall back to local application-default credentials
        pass
    import google.auth
    creds, _ = google.auth.default(scopes=SCOPES)
    return gspread.authorize(creds)


def open_sheet(gc, sheet: str):
    import gspread

    if sheet.startswith("http"):
        return gc.open_by_url(sheet)
    try:
        return gc.open_by_key(sheet)
    except Exception:
        pass
    try:
        return gc.open(sheet)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(sheet)
        print(f"created new Google Sheet '{sheet}': {sh.url}")
        return sh


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not math.isfinite(v) else round(v, 6)
    if isinstance(v, (int, str, bool)):
        return v
    if hasattr(v, "item"):  # numpy scalar
        return _cell(v.item())
    return json.dumps(v, default=str)[:4000]


def _write(sh, title: str, df: pd.DataFrame):
    import gspread

    rows = [[str(c) for c in df.columns]] + [[_cell(v) for v in r] for r in df.itertuples(index=False)]
    n_r, n_c = max(len(rows), 2), max(len(rows[0]), 1)
    try:
        ws = sh.worksheet(title)
        ws.clear()
        if ws.row_count < n_r or ws.col_count < n_c:
            ws.resize(rows=max(ws.row_count, n_r), cols=max(ws.col_count, n_c))
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=n_r, cols=n_c)
    ws.update(range_name="A1", values=rows, value_input_option="RAW")
    try:
        ws.freeze(rows=1, cols=1)
        ws.format("1:1", {"textFormat": {"bold": True}})
    except Exception:
        pass


def _blocks(ds_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(ds_dir, "*", "seed*", "metrics.json"))):
        run = os.path.basename(os.path.dirname(os.path.dirname(f)))
        if run == "tuning":
            continue
        seed = os.path.basename(os.path.dirname(f))
        for r in load_json(f, []) or []:
            if "block" not in r:
                continue
            row = {"run": run, "seed": seed}
            for k, v in r.items():
                if k == "diag":
                    for dk, dv in (v or {}).items():
                        row[f"diag_{dk}"] = dv
                else:
                    row[k] = v
            rows.append(row)
    return pd.DataFrame(rows)


def _first(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    front = [c for c in cols if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


def sync(runs_root: str, dataset: str, sheet: str, reference: str = "tidegr",
         sa_key: str | None = None, gc=None) -> str:
    """Write all results of <runs_root>/<dataset> into the Google Sheet. Returns the sheet URL."""
    ds_dir = os.path.join(runs_root, dataset)
    gc = gc or client(sa_key)
    sh = open_sheet(gc, sheet)
    df = collect(ds_dir)
    if not df.empty:
        df = _first(df.sort_values(["run", "seed_dir"]), ["run", "seed_dir", "method", "acc_ndcg@20",
                                                          "acc_recall@20", "ece", "aurc", "coverage"])
        _write(sh, f"{dataset}_table", table(df).reset_index())
        _write(sh, f"{dataset}_runs", df)
        sig = significance(ds_dir, reference)
        if len(sig):
            _write(sh, f"{dataset}_significance", sig)
    blk = _blocks(ds_dir)
    if not blk.empty:
        _write(sh, f"{dataset}_blocks", _first(blk, ["run", "seed", "block", "block_name", "acc_ndcg@20",
                                                     "acc_recall@20", "ece", "coverage"]))
    st = load_json(os.path.join(ds_dir, "status.json"), {}) or {}
    if st:
        _write(sh, f"{dataset}_status", pd.DataFrame(
            [{"run": k, **{kk: vv for kk, vv in v.items() if kk != "error"},
              "error": (v.get("error") or "")[-500:]} for k, v in st.items()]))
    # append-only sync log
    import gspread
    try:
        lg = sh.worksheet("log")
    except gspread.WorksheetNotFound:
        lg = sh.add_worksheet(title="log", rows=2, cols=4)
        lg.append_row(["time (UTC)", "dataset", "finished run/seeds", "runs_root"])
    lg.append_row([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), dataset, int(len(df)), runs_root])
    for name in ("Sheet1", "Blatt1"):  # remove the empty default tab of a new sheet
        try:
            if len(sh.worksheets()) > 1:
                sh.del_worksheet(sh.worksheet(name))
        except Exception:
            pass
    print(f"[sheets] {dataset}: {len(df)} finished run/seeds, {len(blk)} block rows -> {sh.url}")
    return sh.url
