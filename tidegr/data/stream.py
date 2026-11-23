"""Common dataset format shared by every loader and every method.

Every dataset loader produces a pandas DataFrame of positive interactions with
columns ``user, item, ts, block, reward`` (raw ids allowed), plus block
metadata. :func:`build_stream` turns it into a :class:`StreamData` object:

* contiguous integer ids, assigned in order of first appearance in time;
* one row per (user, item, block) (earliest timestamp, max reward);
* per-block split into train (0), validation (1, time-last ``val_frac``) and
  retention (2, random ``ret_frac`` of the rest, never trained on);
* first-seen block of every user and item;
* optional item side features, random-exposure evaluation records and a
  synthetic preference-swap injection (only for the H3 pilot).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

PART_TRAIN, PART_VAL, PART_RET = 0, 1, 2


@dataclass
class StreamData:
    name: str
    n_users: int
    n_items: int
    # events, sorted by (block, ts)
    ev_user: np.ndarray
    ev_item: np.ndarray
    ev_ts: np.ndarray
    ev_block: np.ndarray
    ev_part: np.ndarray
    ev_reward: np.ndarray
    # blocks
    n_blocks: int
    block_names: list
    block_season: np.ndarray  # season index per block
    n_seasons: int
    warmup_blocks: list  # indices of history (warm-up) blocks
    stream_blocks: list  # indices of stream blocks
    block_ptr: np.ndarray  # events of block b are [block_ptr[b], block_ptr[b+1])
    # first appearance
    user_first_block: np.ndarray
    item_first_block: np.ndarray
    # optional
    item_feat: np.ndarray | None = None  # [n_items, F] float32
    rnd_user: np.ndarray | None = None
    rnd_item: np.ndarray | None = None
    rnd_label: np.ndarray | None = None
    rnd_block: np.ndarray | None = None
    swapped_users: np.ndarray | None = None
    swap_block: int | None = None
    aci_round_seconds: int = 3600
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def block_slice(self, b: int) -> slice:
        return slice(int(self.block_ptr[b]), int(self.block_ptr[b + 1]))

    def block_idx(self, b: int, parts=(PART_TRAIN, PART_VAL, PART_RET)) -> np.ndarray:
        s = self.block_slice(b)
        idx = np.arange(s.start, s.stop)
        if len(parts) == 3:
            return idx
        return idx[np.isin(self.ev_part[s], np.asarray(parts))]

    def pairs(self, b: int, parts=(PART_TRAIN,)):
        idx = self.block_idx(b, parts)
        return self.ev_user[idx], self.ev_item[idx], self.ev_ts[idx], self.ev_reward[idx]

    def history_mask_upto(self, b: int, parts=(PART_TRAIN, PART_VAL, PART_RET)) -> np.ndarray:
        """Boolean mask over events: all events in blocks < b with the given parts."""
        m = np.zeros(len(self.ev_user), dtype=bool)
        m[: int(self.block_ptr[b])] = True
        if len(parts) < 3:
            m &= np.isin(self.ev_part, np.asarray(parts))
        return m

    def item_counts(self, b: int) -> np.ndarray:
        """Train-part interaction counts per item in block b (zeros for b<0)."""
        if b < 0:
            return np.zeros(self.n_items, dtype=np.float32)
        u, i, _, _ = self.pairs(b, (PART_TRAIN,))
        return np.bincount(i, minlength=self.n_items).astype(np.float32)

    def summary(self) -> dict:
        per_block = np.diff(self.block_ptr)
        return {
            "name": self.name,
            "users": int(self.n_users),
            "items": int(self.n_items),
            "events": int(len(self.ev_user)),
            "blocks": int(self.n_blocks),
            "warmup_blocks": len(self.warmup_blocks),
            "stream_blocks": len(self.stream_blocks),
            "events_per_block_min": int(per_block.min()),
            "events_per_block_max": int(per_block.max()),
            "item_feat_dim": None if self.item_feat is None else int(self.item_feat.shape[1]),
            "random_eval_records": None if self.rnd_user is None else int(len(self.rnd_user)),
        }

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict) -> "StreamData":
        return StreamData(**d)


def build_stream(
    df: pd.DataFrame,
    *,
    name: str,
    block_names: list,
    block_season: list,
    n_seasons: int,
    n_warmup: int,
    val_frac: float = 0.1,
    ret_frac: float = 0.05,
    seed: int = 0,
    item_feat_raw: dict | None = None,
    random_df: pd.DataFrame | None = None,
    swap_frac: float = 0.0,
    swap_stream_offset: int = 3,
    aci_round_seconds: int = 3600,
    meta: dict | None = None,
) -> StreamData:
    """Build the common StreamData object.

    df columns: user, item, ts (int seconds), block (0..n_blocks-1), reward (float).
    item_feat_raw: {"ids": raw item ids array, "X": float32 matrix} (optional).
    random_df: columns user, item, ts, block, label (optional; KuaiRand random exposure).
    """
    rng = np.random.default_rng(seed)
    n_blocks = len(block_names)
    df = df[["user", "item", "ts", "block", "reward"]].copy()
    df = df[(df.block >= 0) & (df.block < n_blocks)]
    df.sort_values(["ts"], kind="stable", inplace=True)

    # contiguous ids in order of first appearance
    u_codes, u_uni = pd.factorize(df["user"], sort=False)
    i_codes, i_uni = pd.factorize(df["item"], sort=False)
    df["u"] = u_codes.astype(np.int64)
    df["i"] = i_codes.astype(np.int64)
    n_users, n_items = len(u_uni), len(i_uni)

    # one row per (u, i, block): earliest ts, max reward
    g = df.groupby(["block", "u", "i"], sort=False)
    df = g.agg(ts=("ts", "min"), reward=("reward", "max")).reset_index()
    df.sort_values(["block", "ts"], kind="stable", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # synthetic preference swap (H3 pilot only): permute item ids for chosen users
    swapped_users = None
    swap_block = None
    if swap_frac and swap_frac > 0:
        swap_block = n_warmup + int(swap_stream_offset)
        if swap_block < n_blocks:
            later_users = df.loc[df.block >= swap_block, "u"].unique()
            n_sw = max(1, int(round(swap_frac * len(later_users))))
            swapped_users = np.sort(rng.choice(later_users, size=n_sw, replace=False))
            perm = rng.permutation(n_items)
            sel = (df.block >= swap_block) & df.u.isin(swapped_users)
            df.loc[sel, "i"] = perm[df.loc[sel, "i"].to_numpy()]
            df = df.groupby(["block", "u", "i"], sort=False).agg(
                ts=("ts", "min"), reward=("reward", "max")).reset_index()
            df.sort_values(["block", "ts"], kind="stable", inplace=True)
            df.reset_index(drop=True, inplace=True)

    # parts
    part = np.zeros(len(df), dtype=np.int8)
    blk = df["block"].to_numpy()
    ptr = np.searchsorted(blk, np.arange(n_blocks + 1), side="left").astype(np.int64)
    for b in range(n_blocks):
        s, e = ptr[b], ptr[b + 1]
        n = e - s
        if n == 0:
            continue
        n_val = int(np.floor(val_frac * n))
        if n_val > 0:
            part[e - n_val: e] = 1  # time-last fraction (rows are sorted by ts)
        rest = np.arange(s, e - n_val)
        n_ret = int(np.floor(ret_frac * len(rest)))
        if n_ret > 0:
            part[rng.choice(rest, size=n_ret, replace=False)] = 2

    ev_user = df["u"].to_numpy(np.int64)
    ev_item = df["i"].to_numpy(np.int64)
    ev_ts = df["ts"].to_numpy(np.int64)
    ev_reward = df["reward"].to_numpy(np.float32)

    ufb = np.full(n_users, n_blocks, dtype=np.int64)
    ifb = np.full(n_items, n_blocks, dtype=np.int64)
    np.minimum.at(ufb, ev_user, blk)
    np.minimum.at(ifb, ev_item, blk)

    item_feat = None
    if item_feat_raw is not None:
        X = np.asarray(item_feat_raw["X"], dtype=np.float32)
        pos = pd.Series(np.arange(len(item_feat_raw["ids"])), index=pd.Index(item_feat_raw["ids"]))
        pos = pos[~pos.index.duplicated()]
        idx = pos.reindex(i_uni).to_numpy()
        item_feat = np.zeros((n_items, X.shape[1]), dtype=np.float32)
        ok = ~np.isnan(idx.astype(float))
        item_feat[ok] = X[idx[ok].astype(np.int64)]
        meta = dict(meta or {})
        meta["item_feat_coverage"] = float(ok.mean())

    rnd = {}
    if random_df is not None and len(random_df):
        umap = pd.Series(np.arange(n_users), index=u_uni)
        imap = pd.Series(np.arange(n_items), index=i_uni)
        ru = umap.reindex(random_df["user"].to_numpy()).to_numpy()
        ri = imap.reindex(random_df["item"].to_numpy()).to_numpy()
        ok = ~(np.isnan(ru.astype(float)) | np.isnan(ri.astype(float)))
        rnd = dict(
            rnd_user=ru[ok].astype(np.int64),
            rnd_item=ri[ok].astype(np.int64),
            rnd_label=random_df["label"].to_numpy()[ok].astype(np.int8),
            rnd_block=random_df["block"].to_numpy()[ok].astype(np.int64),
        )

    return StreamData(
        name=name,
        n_users=n_users,
        n_items=n_items,
        ev_user=ev_user,
        ev_item=ev_item,
        ev_ts=ev_ts,
        ev_block=blk.astype(np.int64),
        ev_part=part,
        ev_reward=ev_reward,
        n_blocks=n_blocks,
        block_names=list(block_names),
        block_season=np.asarray(block_season, dtype=np.int64),
        n_seasons=int(n_seasons),
        warmup_blocks=list(range(n_warmup)),
        stream_blocks=list(range(n_warmup, n_blocks)),
        block_ptr=ptr,
        user_first_block=ufb,
        item_first_block=ifb,
        item_feat=item_feat,
        swapped_users=swapped_users,
        swap_block=swap_block,
        aci_round_seconds=int(aci_round_seconds),
        meta=dict(meta or {}),
        **rnd,
    )
