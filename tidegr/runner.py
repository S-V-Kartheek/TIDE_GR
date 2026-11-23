"""Prequential (test-then-train) runner with crash-safe, cross-account resume.

For every stream block k:
  1 TEST     on block k with the model trained up to block k-1 (full ranking)
  2 OBSERVE  drift detection + RL reward (method.observe_test)
  3 UPDATE   learn from block k's train split (method.update)
  4 CALIBRATE on block k's validation split
  5 FINALIZE add block k's validation edges to the past
  6 RETAIN   evaluate retention slices of all past stream blocks (never trained on)
  7 CHECKPOINT (atomic) -> next block

A checkpoint holds the method state, calibrator, ACI, metrics so far and all
RNG states, so a run continues on any machine/account from the last finished
block (or the last saved warm-up epoch).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .calibration import ACI, Calibrator
from .config import cfg_hash, to_plain
from .data.stream import PART_RET, PART_TRAIN, PART_VAL, StreamData
from .eval.metrics import CSR, aurc, auc_score, brier_score, ece_score, nll_score, rank_eval
from .models import build_method
from .utils import (TimeBudget, TimeLimitReached, atomic_json_save, atomic_torch_save, get_rng_state,
                    log, safe_torch_load, set_log_file, set_rng_state)


class Context:
    def __init__(self, runner):
        self.r = runner
        self.budget = runner.budget

    def save(self):
        self.r.save()

    def val_metric(self, method, block) -> float:
        return self.r.val_ndcg(method, block)

    def memory_ndcg(self, method, block) -> float:
        u, i = method.memory.val_pairs()
        if len(u) == 0:
            return 0.0
        res = self.r.eval_targets(method, block, u, i, include_train_of_block=True, light=True,
                                  cache_key=("mem", block))  # same targets before/after the update
        return float(np.mean(res["acc"]["ndcg@20"])) if res else 0.0


class Runner:
    def __init__(self, cfg, data: StreamData, run_dir: str, seed: int = 0, device=None,
                 max_hours: float | None = None, verbose: bool = True):
        self.cfg, self.data, self.run_dir, self.seed = cfg, data, run_dir, int(seed)
        self.device = device or torch.device("cpu")
        self.budget = TimeBudget(max_hours)
        self.verbose = verbose
        os.makedirs(run_dir, exist_ok=True)
        set_log_file(os.path.join(run_dir, "log.txt"))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed + 12345)
        self.method = build_method(cfg, data, self.device, self.seed)
        ev = cfg.get("eval", {}) or {}
        self.K = int(ev.get("k", 20))
        self.n_neg_cal = int(ev.get("n_neg_calib", 99))
        self.max_pos_per_user = int(ev.get("max_pos_per_user", 50))
        self.val_max_users = int(ev.get("val_max_users", 3000))
        self.chunk = int(ev.get("chunk", 1024))
        self.use_var = bool(cfg.method.get("calib_use_var", True)) and self.method.supports_var
        self.cal = Calibrator(use_var=self.use_var)
        a = cfg.get("aci", {}) or {}
        self.aci = ACI(alpha=0.5, gamma=float(a.get("gamma", 0.05)),
                       delay=int(a.get("delay", 1)), tau0=float(a.get("tau0", 0.0)))
        self.phase = "init"
        self.next_idx = 0
        self.records: list = []
        self.retention: dict = {}  # eval_block -> {slice_block: recall@20}
        self.block_times: list = []
        self.wall = 0.0
        self.cfg_hash = cfg_hash(cfg)
        # model-independent "seen" bookkeeping (train/val splits only)
        tv = np.isin(data.ev_part, (PART_TRAIN, PART_VAL))
        self.item_first_tv = np.full(data.n_items, data.n_blocks, np.int64)
        np.minimum.at(self.item_first_tv, data.ev_item[tv], data.ev_block[tv])
        self.user_first_tv = np.full(data.n_users, data.n_blocks, np.int64)
        np.minimum.at(self.user_first_tv, data.ev_user[tv], data.ev_block[tv])
        trn = data.ev_part == PART_TRAIN
        self.item_first_train = np.full(data.n_items, data.n_blocks, np.int64)
        np.minimum.at(self.item_first_train, data.ev_item[trn], data.ev_block[trn])
        self._val_cache: dict = {}
        self._eval_cache: dict = {}

    # ================================================================== checkpoint
    @property
    def ckpt(self):
        return os.path.join(self.run_dir, "state.pt")

    def save(self):
        st = {
            "phase": self.phase, "next_idx": self.next_idx, "records": self.records,
            "retention": self.retention, "block_times": self.block_times,
            "cal": self.cal.state_dict(), "aci": self.aci.state_dict(),
            "method": self.method.state_dict(), "rng": get_rng_state(self.rng),
            "wall": self.wall + self.budget.elapsed(), "cfg_hash": self.cfg_hash,
            "cfg": to_plain(self.cfg), "saved_at": time.time(),
        }
        atomic_torch_save(st, self.ckpt)

    def load(self) -> bool:
        st = safe_torch_load(self.ckpt)
        if st is None:
            return False
        if st.get("cfg_hash") != self.cfg_hash:
            log(f"[resume] WARNING: config changed since the checkpoint was written "
                f"({st.get('cfg_hash')} -> {self.cfg_hash}); resuming anyway")
        self.phase, self.next_idx = st["phase"], st["next_idx"]
        self.records, self.retention = st["records"], st["retention"]
        self.block_times = st["block_times"]
        self.cal.load_state_dict(st["cal"])
        self.aci.load_state_dict(st["aci"])
        self.method.load_state_dict(st["method"])
        set_rng_state(st["rng"], self.rng)
        self.wall = st.get("wall", 0.0)
        log(f"[resume] {self.run_dir}: phase={self.phase} next stream block index={self.next_idx}")
        return True

    # ================================================================== helpers
    def _known_items(self, block: int, include_train_of_block: bool) -> torch.Tensor:
        known = self.item_first_tv < block
        if include_train_of_block:
            known |= self.item_first_train <= block
        return torch.as_tensor(known, device=self.device)

    def _history_keys(self, block: int, include_train_of_block: bool) -> np.ndarray:
        d = self.data
        end = int(d.block_ptr[block])
        keys = d.ev_user[:end] * d.n_items + d.ev_item[:end]
        if include_train_of_block:
            s = d.block_slice(block)
            tr = d.ev_part[s] == PART_TRAIN
            keys = np.concatenate([keys, d.ev_user[s][tr] * d.n_items + d.ev_item[s][tr]])
        return keys

    def _history(self, users: np.ndarray, block: int, include_train_of_block: bool,
                 exclude_keys: np.ndarray | None = None) -> CSR:
        d = self.data
        end = int(d.block_ptr[block])
        u, i = d.ev_user[:end], d.ev_item[:end]
        if include_train_of_block:
            s = d.block_slice(block)
            tr = d.ev_part[s] == PART_TRAIN
            u = np.concatenate([u, d.ev_user[s][tr]])
            i = np.concatenate([i, d.ev_item[s][tr]])
        if exclude_keys is not None and len(exclude_keys):
            keep = ~np.isin(u * d.n_items + i, exclude_keys)
            u, i = u[keep], i[keep]
        return CSR.build(users, u, i, d.n_users)

    def eval_targets(self, method, block, tu, ti, include_train_of_block=False, light=True,
                     exclude_target_from_mask=True, extra=False, return_top=0, cache_key=None):
        """Full-ranking evaluation of explicit targets at `block`."""
        prep = self._eval_cache.get(cache_key) if cache_key is not None else None
        if prep is None:
            prep = self._prepare_targets(block, tu, ti, include_train_of_block, exclude_target_from_mask)
            if cache_key is not None:
                if len(self._eval_cache) > 8:
                    self._eval_cache.clear()
                self._eval_cache[cache_key] = prep
        if prep is None:
            return None
        users, targets, mask, known = prep
        method.prepare_eval(block, light=light)
        extra_fns = {}
        if extra and method.supports_rel:
            extra_fns["rel"] = method.score_rel
        if extra and method.supports_ts:
            extra_fns["ts"] = method.score_ts
        res = rank_eval(method.score, users, targets, mask, known, ks=(10, self.K), chunk=self.chunk,
                        device=self.device, return_top=return_top, extra_score_fns=extra_fns)
        res["users"] = users
        res["targets"] = targets
        return res

    def _prepare_targets(self, block, tu, ti, include_train_of_block, exclude_target_from_mask):
        known = self._known_items(block, include_train_of_block)
        kn = known.cpu().numpy()
        ok = kn[ti]
        tu, ti = tu[ok], ti[ok]
        if len(tu) == 0:
            return None
        tkeys = tu * self.data.n_items + ti
        if not exclude_target_from_mask:
            # targets already consumed before are masked, so drop them (new-item recommendation)
            seen = np.isin(tkeys, self._history_keys(block, include_train_of_block))
            tu, ti, tkeys = tu[~seen], ti[~seen], tkeys[~seen]
            if len(tu) == 0:
                return None
        users = np.unique(tu)
        keys = tkeys if exclude_target_from_mask else None
        mask = self._history(users, block, include_train_of_block, exclude_keys=keys)
        targets = CSR.build(users, tu, ti, self.data.n_users)
        return users, targets, mask, known

    def val_ndcg(self, method, block) -> float:
        if block not in self._val_cache:
            u, i, _, _ = self.data.pairs(block, (PART_VAL,))
            users = np.unique(u)
            if len(users) > self.val_max_users:
                r = np.random.default_rng(1000 + block)
                users = np.sort(r.choice(users, self.val_max_users, replace=False))
                keep = np.isin(u, users)
                u, i = u[keep], i[keep]
            self._val_cache[block] = (u, i)
        u, i = self._val_cache[block]
        res = self.eval_targets(method, block, u, i, include_train_of_block=True, light=True,
                                exclude_target_from_mask=False, cache_key=("val", block))
        return float(np.mean(res["acc"][f"ndcg@{self.K}"])) if res else 0.0

    def _calib_pairs(self, tu, ti, block, include_train_of_block):
        """Positives (capped per user) + n_neg uniformly sampled known items each."""
        known = np.where(self._known_items(block, include_train_of_block).cpu().numpy())[0]
        ok = np.isin(ti, known)
        tu, ti = tu[ok], ti[ok]
        if len(tu) == 0:
            return None
        order = self.rng.permutation(len(tu))
        tu, ti = tu[order], ti[order]
        srt = np.argsort(tu, kind="stable")
        tu, ti = tu[srt], ti[srt]
        first = np.r_[0, np.flatnonzero(np.diff(tu)) + 1]
        rank = np.arange(len(tu)) - np.repeat(first, np.diff(np.r_[first, len(tu)]))
        keep = rank < self.max_pos_per_user
        tu, ti = tu[keep], ti[keep]
        nu = np.repeat(tu, self.n_neg_cal)
        ni = known[self.rng.integers(0, len(known), len(nu))]
        pu = np.concatenate([tu, nu])
        pi = np.concatenate([ti, ni])
        y = np.concatenate([np.ones(len(tu)), np.zeros(len(nu))]).astype(np.float32)
        return pu, pi, y

    def _fit_calibration(self, block):
        u, i, _, _ = self.data.pairs(block, (PART_VAL,))
        cp = self._calib_pairs(u, i, block, include_train_of_block=True)
        if cp is None:
            return
        self.method.prepare_eval(block, light=False)
        m, v = self.method.pair_mv(cp[0], cp[1])
        self.cal.fit(m, v if self.use_var else None, cp[2])
        if hasattr(self.method, "var_scale"):
            self.method.var_scale = self.cal.nu  # reliability ranking in calibrated units

    # ================================================================== warm-up
    def _warmup(self):
        d = self.data
        ctx = Context(self)
        t0 = time.time()
        info = self.method.warmup(ctx)
        last = d.warmup_blocks[-1]
        self._fit_calibration(last)
        # ACI target alpha: selective error at 50% coverage on warm-up validation
        u, i, _, _ = d.pairs(last, (PART_VAL,))
        res = self.eval_targets(self.method, last, u, i, include_train_of_block=True, light=False,
                                return_top=self.K)
        if res is not None:
            conf = self._confidence(res["users"], res["acc"]["top"])
            err = (res["acc"][f"hits@{self.K}"] == 0).astype(float)
            order = np.argsort(-conf)
            half = order[: max(1, len(order) // 2)]
            self.aci.alpha = float(err[half].mean())
            self.aci.tau = float(np.median(conf))
            log(f"[aci] alpha (selective error at 50% coverage on warm-up val) = {self.aci.alpha:.3f}")
        self.method.finalize_block(last)
        self.records.append({"phase": "warmup", "time_s": time.time() - t0,
                             "calib": self.cal.state_dict(), **{f"warmup_{k}": v for k, v in (info or {}).items()
                                                                 if not isinstance(v, (list, dict))}})
        self.phase = "stream"
        self.next_idx = 0
        self.save()

    def _confidence(self, users, top):
        uu = np.repeat(users, top.shape[1])
        m, v = self.method.pair_mv(uu, top.reshape(-1))
        p = self.cal.predict(m, v if self.use_var else None)
        return p.reshape(len(users), -1).mean(1)

    # ================================================================== one stream block
    def _test(self, k: int) -> dict:
        d = self.data
        rec: dict = {"block": int(k), "block_name": d.block_names[k]}
        self._last_test = {}
        u, i, ts, _ = d.pairs(k, (PART_TRAIN, PART_VAL, PART_RET))
        ok = (self.user_first_tv[u] < k)
        u, i, ts = u[ok], i[ok], ts[ok]
        t0 = time.time()
        res = self.eval_targets(self.method, k, u, i, include_train_of_block=False, light=False,
                                exclude_target_from_mask=False, extra=True, return_top=self.K)
        if res is None:
            return rec
        users = res["users"]
        for mode in res:
            if mode in ("users", "targets"):
                continue
            for key, arr in res[mode].items():
                if key != "top":
                    rec[f"{mode}_{key}"] = float(np.mean(arr))
        rec["n_users"] = int(len(users))
        rec["test_s"] = time.time() - t0
        acc = res["acc"]
        # calibration on positives + sampled negatives (same model, before training on block k)
        tl = res["targets"]
        tu_ = np.repeat(users, tl.lengths())
        cp = self._calib_pairs(tu_, tl.indices, k, include_train_of_block=False)
        pair_info = None
        if cp is not None:
            m, v = self.method.pair_mv(cp[0], cp[1])
            p = self.cal.predict(m, v if self.use_var else None)
            y = cp[2]
            rec.update({"ece": ece_score(p, y), "nll": nll_score(p, y), "brier": brier_score(p, y)})
            if v is not None and self.use_var:
                eu = self.cal.epistemic(m, v)
                rec["epistemic_share"] = float(eu.mean() / max((p * (1 - p)).mean(), 1e-12))
            pair_info = (cp[0], cp[1], y, p)
        # selective recommendation + ACI
        conf = self._confidence(users, acc["top"])
        err = (acc[f"hits@{self.K}"] == 0).astype(float)
        rec["aurc"] = aurc(conf, err)
        first_ts = np.full(d.n_users, np.iinfo(np.int64).max)
        np.minimum.at(first_ts, u, ts)
        bstart = ts.min() if len(ts) else 0
        rounds = (first_ts[users] - bstart) // max(d.aci_round_seconds, 1)
        served = self.aci.run_block(rounds, conf, err)
        rec["coverage"] = float(served.mean())
        rec["selective_err"] = float(err[served].mean()) if served.any() else 0.0
        rec["all_err"] = float(err.mean())
        rec.update(self.aci.long_run())
        # open world: users / items first seen in the previous block
        nu = self.user_first_tv[users] == k - 1
        rec["new_user_recall@20"] = float(acc[f"recall@{self.K}"][nu].mean()) if nu.any() else None
        rec["n_new_users"] = int(nu.sum())
        new_item = self.item_first_tv == k - 1
        top = acc["top"]
        hits, tot = 0, 0
        for r in range(len(users)):
            t = tl.row(r)
            tn = t[new_item[t]]
            if len(tn):
                tot += len(tn)
                hits += int(np.isin(top[r], tn).sum())
        rec["new_item_recall@20"] = hits / tot if tot else None
        rec["n_new_item_targets"] = int(tot)
        # unbiased evaluation on random exposure (KuaiRand)
        if d.rnd_user is not None:
            rec.update(self._unbiased(k))
        # per-user arrays for significance tests
        os.makedirs(os.path.join(self.run_dir, "per_user"), exist_ok=True)
        np.savez_compressed(os.path.join(self.run_dir, "per_user", f"block_{k}.npz"), users=users,
                            ndcg=acc[f"ndcg@{self.K}"], recall=acc[f"recall@{self.K}"])
        self._last_test = {"users": users, "ndcg": acc[f"ndcg@{self.K}"], "ece": rec.get("ece", 0.0)}
        if pair_info is not None:
            self._last_test.update(pair_u=pair_info[0], pair_i=pair_info[1], pair_y=pair_info[2],
                                   pair_p=pair_info[3])
        return rec

    @torch.no_grad()
    def _unbiased(self, k):
        d = self.data
        m = d.rnd_block == k
        ru, ri, rl = d.rnd_user[m], d.rnd_item[m], d.rnd_label[m]
        known = self._known_items(k, False).cpu().numpy()
        ok = (self.user_first_tv[ru] < k) & known[ri]
        ru, ri, rl = ru[ok], ri[ok], rl[ok]
        out = {}
        if len(ru) == 0:
            return out
        fns = {"acc": self.method.score}
        if self.method.supports_ts:
            fns["ts"] = self.method.score_ts
        users = np.unique(ru)
        pos = np.searchsorted(users, ru)
        for name, fn in fns.items():
            sc = np.zeros(len(ru), np.float32)
            for a in range(0, len(users), self.chunk):
                b = min(len(users), a + self.chunk)
                sel = (pos >= a) & (pos < b)
                S = fn(torch.as_tensor(users[a:b], device=self.device))
                sc[sel] = S[torch.as_tensor(pos[sel] - a, device=self.device),
                            torch.as_tensor(ri[sel], device=self.device)].float().cpu().numpy()
            aucs, ndcgs = [], []
            order = np.argsort(pos, kind="stable")
            bounds = np.r_[0, np.flatnonzero(np.diff(pos[order])) + 1, len(order)]
            for s, e in zip(bounds[:-1], bounds[1:]):
                idx = order[s:e]
                lab = rl[idx]
                if lab.min() == lab.max():
                    continue
                aucs.append(auc_score(sc[idx], lab))
                o = np.argsort(-sc[idx])[:10]
                dcg = (lab[o] / np.log2(np.arange(2, len(o) + 2))).sum()
                ideal = (np.sort(lab)[::-1][:10] / np.log2(np.arange(2, min(10, len(lab)) + 2))).sum()
                ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
            out[f"unbiased_{name}_auc"] = float(np.mean(aucs)) if aucs else None
            out[f"unbiased_{name}_ndcg@10"] = float(np.mean(ndcgs)) if ndcgs else None
            out["unbiased_users"] = len(aucs)
        return out

    def _drift_auc(self, k, rec):
        d = self.data
        if d.swapped_users is None or d.swap_block is None or k < d.swap_block:
            return
        D = getattr(getattr(self.method, "drift", None), "D", None)
        if D is None:
            return
        users = self._last_test["users"]
        lab = np.isin(users, d.swapped_users).astype(int)
        rec["drift_auc"] = auc_score(D[users], lab)
        end = int(d.block_ptr[k])
        act = np.bincount(d.ev_user[:end], minlength=d.n_users)[users]
        low = act <= np.median(act)
        rec["drift_auc_lowactivity"] = auc_score(D[users][low], lab[low])
        rec["drift_auc_highactivity"] = auc_score(D[users][~low], lab[~low])

    def _retention(self, k):
        """Recall@20 on held-out retention slices of every stream block <= k
        (evaluated right after learning block k; slices are never trained on)."""
        d = self.data
        out = {}
        for j in d.stream_blocks:
            if j > k:
                break
            u, i, _, _ = d.pairs(j, (PART_RET,))
            ok = self.user_first_tv[u] <= j
            if not ok.any():
                continue
            res = self.eval_targets(self.method, k, u[ok], i[ok], include_train_of_block=True,
                                    light=True, exclude_target_from_mask=True)
            if res:
                out[int(j)] = float(np.mean(res["acc"][f"recall@{self.K}"]))
        self.retention[int(k)] = out
        return out

    def _stream_block(self, k: int):
        t0 = time.time()
        rec = self._test(k)
        if "pair_u" in self._last_test:
            self.method.observe_test(k, self._last_test)
            self._drift_auc(k, rec)
        ctx = Context(self)
        tu = time.time()
        info = self.method.update(k, ctx) or {}
        rec["update_s"] = time.time() - tu
        rec["train_epochs"] = info.get("epochs")
        self._fit_calibration(k)
        self.method.finalize_block(k)
        ret = self._retention(k)
        rec["retention_mean_recall@20"] = float(np.mean(list(ret.values()))) if ret else None
        diag = self.method.diagnostics()
        rec["diag"] = {kk: vv for kk, vv in diag.items() if not isinstance(vv, np.ndarray)}
        rec["block_s"] = time.time() - t0
        return rec

    # ================================================================== main loop
    def run(self, resume: bool = True) -> dict:
        if resume:
            self.load()
        if self.phase == "done":
            log(f"[run] already finished: {self.run_dir}")
            return self.summary()
        try:
            if self.phase == "init":
                log(f"[run] warm-up on {len(self.data.warmup_blocks)} history blocks")
                self.phase = "warmup"
                self._warmup()
            elif self.phase == "warmup":
                log("[run] resuming warm-up")
                self._warmup()
            blocks = self.data.stream_blocks
            while self.next_idx < len(blocks):
                k = blocks[self.next_idx]
                est = float(np.mean(self.block_times[-3:])) if self.block_times else 0.0
                if not self.budget.ok(need_s=1.5 * est + 0.5 * float(self.cfg.get("budget_margin_s", 120))):
                    raise TimeLimitReached("time budget reached before block")
                rec = self._stream_block(k)
                self.block_times.append(rec["block_s"])
                self.records.append(rec)
                self.next_idx += 1
                self.save()
                self._log_block(rec)
            self.phase = "done"
            self.save()
            summ = self.summary()
            atomic_json_save(summ, os.path.join(self.run_dir, "summary.json"))
            atomic_json_save(self.records, os.path.join(self.run_dir, "metrics.json"))
            log(f"[run] finished {self.run_dir}")
            return summ
        except TimeLimitReached as e:
            log(f"[run] {e}. Progress is saved; run the same command again (any account) to resume.")
            raise

    def _log_block(self, r):
        keys = ["acc_recall@20", "acc_ndcg@20", "ece", "aurc", "coverage", "retention_mean_recall@20",
                "unbiased_acc_auc", "drift_auc", "block_s"]
        msg = " ".join(f"{k}={r[k]:.4f}" for k in keys if isinstance(r.get(k), (int, float)))
        arms = r.get("diag", {}).get("arms")
        log(f"[block {r['block']} {r['block_name']}] users={r.get('n_users')} {msg}"
            + (f" arms={arms}" if arms else ""))

    def summary(self) -> dict:
        recs = [r for r in self.records if "block" in r]
        out: dict = {"run_dir": self.run_dir, "seed": self.seed, "n_blocks": len(recs),
                     "method": self.cfg.method.get("name", self.cfg.method.get("kind"))}
        if not recs:
            return out
        keys = sorted({k for r in recs for k, v in r.items() if isinstance(v, (int, float)) and k != "block"})
        for k in keys:
            vals = [r[k] for r in recs if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])]
            if vals:
                out[k] = float(np.mean(vals))
        # backward transfer over stream blocks
        stream = [b for b in self.data.stream_blocks if b in self.retention]
        if stream:
            last = stream[-1]
            diffs = [self.retention[last][j] - self.retention[j][j] for j in stream[:-1]
                     if j in self.retention[last] and j in self.retention.get(j, {})]
            out["bwt_recall@20"] = float(np.mean(diffs)) if diffs else None
            out["final_retention_recall@20"] = float(np.mean(list(self.retention[last].values())))
        out.update({k: v for k, v in self.aci.long_run().items()})
        out["wall_hours"] = (self.wall + self.budget.elapsed()) / 3600
        return out
