"""KuaiRand-Pure loader (Kuaishou short-video app; Gao et al., CIKM 2022).

* History (warm-up): standard-feed log Apr 8 - Apr 21 2022 (one block per day).
* Stream: standard-feed log Apr 22 - May 8 2022 (one block per day).
* Random-exposure log Apr 22 - May 8: used ONLY for unbiased evaluation.
* Positive (default "click_engaged"): is_click=1 and (long_view or like or
  follow or comment or forward) and is_hate=0.
* Reward (for the RL Q-head ablation): 1 + long_view + 2*like + 3*follow
  + 2*comment + 3*forward.
* Item side features: video_features_basic_pure.csv (categorical one-hots,
  tag multi-hot, log size/duration). Captions are optional (3.2 GB download).
"""
from __future__ import annotations

import glob
import os
import tarfile

import numpy as np
import pandas as pd

from ..utils import log
from .download import download
from .stream import build_stream

URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz?download=1"
MD5 = "0820331067a3784d9691136f772b35a7"
CAPTIONS_URL = "https://zenodo.org/records/18159199/files/kuairand_video_captions.csv?download=1"

LOG_COLS = ["user_id", "video_id", "date", "time_ms", "is_click", "is_like", "is_follow",
            "is_comment", "is_forward", "is_hate", "long_view"]


def _find(root: str, name: str) -> str | None:
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    return hits[0] if hits else None


def fetch(raw_dir: str) -> str:
    data_dir = os.path.join(raw_dir, "kuairand")
    if _find(data_dir, "log_standard_4_08_to_4_21_pure.csv"):
        return data_dir
    tgz = download(URL, os.path.join(raw_dir, "KuaiRand-Pure.tar.gz"), md5=MD5)
    log("[kuairand] extracting archive")
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(data_dir)
    return data_dir


def _read_log(path: str) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=5)
    cols = [c for c in LOG_COLS if c in head.columns]
    missing = set(LOG_COLS) - set(cols)
    if missing:
        log(f"[kuairand] columns missing in {os.path.basename(path)}: {sorted(missing)}")
    df = pd.read_csv(path, usecols=cols)
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = 0
    return df


def _positive(df: pd.DataFrame, rule: str) -> np.ndarray:
    click = df["is_click"].to_numpy() == 1
    engaged = ((df["long_view"] == 1) | (df["is_like"] == 1) | (df["is_follow"] == 1)
               | (df["is_comment"] == 1) | (df["is_forward"] == 1)).to_numpy()
    not_hate = df["is_hate"].to_numpy() != 1
    if rule == "click":
        return click & not_hate
    if rule == "long_view":
        return (df["long_view"].to_numpy() == 1) & not_hate
    return click & engaged & not_hate  # click_engaged (default)


def _reward(df: pd.DataFrame) -> np.ndarray:
    return (1.0 + df["long_view"] + 2 * df["is_like"] + 3 * df["is_follow"]
            + 2 * df["is_comment"] + 3 * df["is_forward"]).to_numpy(np.float32)


def _item_features(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        log("[kuairand] video_features_basic_pure.csv not found: no item features")
        return None
    v = pd.read_csv(path)
    feats = []
    for c in ["video_type", "upload_type", "visible_status", "music_type"]:
        if c in v.columns:
            s = v[c].astype(str).fillna("NA")
            top = s.value_counts().index[:50]
            s = s.where(s.isin(top), "OTHER")
            feats.append(pd.get_dummies(s, prefix=c).astype(np.float32))
    if "tag" in v.columns:
        tags = v["tag"].fillna("").astype(str).str.replace(" ", "")
        dummies = tags.str.get_dummies(sep=",")
        top = dummies.sum().sort_values(ascending=False).index[:100]
        feats.append(dummies[top].add_prefix("tag_").astype(np.float32))
    num = []
    for c in ["video_duration", "server_width", "server_height"]:
        if c in v.columns:
            x = np.log1p(pd.to_numeric(v[c], errors="coerce").fillna(0).clip(lower=0))
            num.append(((x - x.mean()) / (x.std() + 1e-6)).rename(c))
    if num:
        feats.append(pd.concat(num, axis=1).astype(np.float32))
    if not feats:
        return None
    X = pd.concat(feats, axis=1).to_numpy(np.float32)
    return {"ids": v["video_id"].to_numpy(), "X": X}


def _captions(raw_dir: str, video_ids: np.ndarray, model_name: str, device: str) -> dict | None:
    path = download(CAPTIONS_URL, os.path.join(raw_dir, "kuairand_video_captions.csv"))
    keep = set(int(x) for x in video_ids)
    parts = []
    for ch in pd.read_csv(path, usecols=["final_video_id", "caption"], chunksize=2_000_000):
        parts.append(ch[ch.final_video_id.isin(keep)])
    cap = pd.concat(parts).drop_duplicates("final_video_id")
    from .text import embed_texts
    X = embed_texts(cap["caption"].fillna("").astype(str).tolist(), model_name, device)
    return {"ids": cap["final_video_id"].to_numpy(), "X": X}


def load(cfg, raw_dir: str, seed: int = 0):
    dcfg = cfg.data
    data_dir = fetch(raw_dir)
    f_hist = _find(data_dir, "log_standard_4_08_to_4_21_pure.csv")
    f_strm = _find(data_dir, "log_standard_4_22_to_5_08_pure.csv")
    f_rand = _find(data_dir, "log_random_4_22_to_5_08_pure.csv")
    log("[kuairand] reading logs")
    std = pd.concat([_read_log(f_hist), _read_log(f_strm)], ignore_index=True)
    pos = _positive(std, dcfg.get("positive", "click_engaged"))
    df = pd.DataFrame({
        "user": std["user_id"].to_numpy(),
        "item": std["video_id"].to_numpy(),
        "ts": (std["time_ms"].to_numpy() // 1000).astype(np.int64),
        "date": std["date"].to_numpy(),
        "reward": _reward(std),
    })[pos]
    log(f"[kuairand] standard log rows={len(std):,} positives={len(df):,} "
        f"({100 * pos.mean():.1f}%)")

    dates = sorted(int(d) for d in np.unique(df["date"]))
    date2block = {d: b for b, d in enumerate(dates)}
    df["block"] = df["date"].map(date2block).astype(np.int64)
    split = int(dcfg.get("stream_start_date", 20220422))
    n_warmup = sum(1 for d in dates if d < split)
    seasons = [int(pd.to_datetime(str(d), format="%Y%m%d").dayofweek) for d in dates]

    random_df = None
    if f_rand and dcfg.get("use_random_eval", True):
        rnd = _read_log(f_rand)
        rnd = rnd[rnd["date"].isin(date2block.keys())]
        random_df = pd.DataFrame({
            "user": rnd["user_id"].to_numpy(),
            "item": rnd["video_id"].to_numpy(),
            "ts": (rnd["time_ms"].to_numpy() // 1000).astype(np.int64),
            "block": rnd["date"].map(date2block).to_numpy(np.int64),
            "label": _positive(rnd, dcfg.get("positive", "click_engaged")).astype(np.int8),
        })

    feat = _item_features(_find(data_dir, "video_features_basic_pure.csv"))
    if dcfg.get("captions", False):
        cap = _captions(raw_dir, df["item"].unique(), dcfg.get("text_model",
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                        "cuda")
        if cap is not None and feat is not None:
            capmap = pd.Series(range(len(cap["ids"])), index=cap["ids"])
            idx = capmap.reindex(feat["ids"]).to_numpy()
            ok = ~np.isnan(idx.astype(float))
            C = np.zeros((len(feat["ids"]), cap["X"].shape[1]), np.float32)
            C[ok] = cap["X"][idx[ok].astype(np.int64)]
            feat = {"ids": feat["ids"], "X": np.concatenate([feat["X"], C], axis=1)}
        elif cap is not None:
            feat = cap

    return build_stream(
        df[["user", "item", "ts", "block", "reward"]],
        name="kuairand",
        block_names=[str(d) for d in dates],
        block_season=seasons,
        n_seasons=7,
        n_warmup=n_warmup,
        val_frac=float(dcfg.get("val_frac", 0.1)),
        ret_frac=float(dcfg.get("ret_frac", 0.05)),
        seed=seed,
        item_feat_raw=feat,
        random_df=random_df,
        swap_frac=float(dcfg.get("swap_frac", 0.0)),
        swap_stream_offset=int(dcfg.get("swap_stream_offset", 3)),
        aci_round_seconds=int(dcfg.get("aci_round_seconds", 3600)),
        meta={"positive_rule": dcfg.get("positive", "click_engaged")},
    )
