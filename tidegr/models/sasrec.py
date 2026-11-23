"""SASRec (Kang & McAuley, ICDM 2018) and SQN (Xin et al., SIGIR 2020) baselines,
fine-tuned block by block under the same prequential protocol.

* Loss: sampled softmax with the same negatives as GraphRec (stronger than the
  original 1-negative BCE, so the baseline is not weakened).
* SQN: an extra Q-learning head trained with the logged reward (KuaiRand:
  weighted feedback signals; Amazon: 1); ranking uses the supervised head,
  as in the SQN paper.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.stream import PART_TRAIN, PART_VAL
from ..features import FeatureBank
from ..utils import TimeLimitReached, log
from .base import Method


class SASRecNet(nn.Module):
    def __init__(self, n_items, d=64, max_len=50, n_layers=2, n_heads=1, dropout=0.2):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.max_len = max_len
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.normal_(self.pos.weight, std=0.05)

    def forward(self, seq):  # seq [B, L] item ids + 1, 0 = pad (left padded)
        B, L = seq.shape
        x = self.item(seq) * (self.item.embedding_dim ** 0.5) + self.pos.weight[:L].unsqueeze(0)
        x = self.drop(x)
        pad = seq == 0
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), 1)
        allpad = pad.all(1)
        pad = pad.clone()
        pad[allpad, -1] = False  # avoid NaN for empty sequences
        h = self.enc(x, mask=causal, src_key_padding_mask=pad)
        return self.ln(h[:, -1])

    def item_emb(self):
        return self.item.weight[1:]


class SeqRec(Method):
    def __init__(self, cfg, data, device, seed):
        super().__init__(cfg, data, device, seed)
        m = cfg.method
        self.mc = m
        self.sqn = m.get("kind") == "sqn"
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.nU, self.nI = data.n_users, data.n_items
        self.L = int(m.get("max_len", 50))
        self.net = SASRecNet(self.nI, int(m.get("dim", 64)), self.L, int(m.get("layers", 2)),
                             int(m.get("heads", 1)), float(m.get("dropout", 0.2))).to(device)
        self.qhead = nn.Linear(int(m.get("dim", 64)), self.nI).to(device) if self.sqn else None
        self.q_gamma = float(m.get("q_gamma", 0.5))
        self.q_weight = float(m.get("q_weight", 1.0))
        self.n_neg = int(m.get("n_neg", 64))
        self.tau_s = float(m.get("tau_s", 1.0))
        self.lr = float(m.get("lr", 1e-3))
        self.batch = int(m.get("batch", 512))
        ep = m.get("epochs", {}) or {}
        self.ep_w_max = int(ep.get("warmup_max", 100))
        self.ep_w_pat = int(ep.get("warmup_patience", 3))
        self.ep_w_every = int(ep.get("warmup_eval_every", 1))
        self.ep_s = int(ep.get("stream", 10))
        self.ep_s_pat = int(ep.get("stream_patience", 2))
        self.fb = FeatureBank(data)
        self.seen_item = np.zeros(self.nI, bool)
        self.train_state = None
        self.cache = {}
        self._training = False
        self.diag = {}

    # ------------------------------------------------------------------ data
    def _targets(self, blocks_train: list, history_upto: int):
        """Target events (train split of blocks_train) with their preceding item sequences."""
        d = self.data
        end = int(d.block_ptr[history_upto + 1])
        idx = np.arange(end)
        b = d.ev_block[:end]
        part = d.ev_part[:end]
        is_tgt = np.isin(b, blocks_train) & (part == PART_TRAIN)
        in_hist = (part == PART_TRAIN) | ((part == PART_VAL) & (b < max(blocks_train)))
        idx = idx[in_hist | is_tgt]
        u, i, t = d.ev_user[idx], d.ev_item[idx], d.ev_ts[idx]
        tgt = is_tgt[idx]
        rw = d.ev_reward[idx]
        order = np.lexsort((t, u))
        u, i, tgt, rw = u[order], i[order], tgt[order], rw[order]
        start = np.zeros(self.nU + 1, np.int64)
        np.add.at(start, u + 1, 1)
        start = np.cumsum(start)
        j = np.where(tgt)[0]
        pos = j[:, None] - self.L + np.arange(self.L)[None, :]
        valid = pos >= start[u[j]][:, None]
        seq = np.where(valid, i[np.clip(pos, 0, None)] + 1, 0)
        return u[j], seq, i[j], rw[j]

    def _seq_eval(self, k):
        R = self.fb.recent(k, self.L)
        return torch.as_tensor(np.where(R >= 0, R + 1, 0), device=self.device)

    # ------------------------------------------------------------------ training
    def _loss(self, seq, pos, rew):
        h = self.net(seq)
        E = self.net.item_emb()
        known = self._known
        neg = known[torch.randint(len(known), (len(pos), self.n_neg), device=self.device)]
        sp = (h * E[pos]).sum(-1)
        sn = torch.einsum("bd,bnd->bn", h, E[neg])
        logits = torch.cat([sp.unsqueeze(1), sn], 1) / self.tau_s
        loss = -F.log_softmax(logits, 1)[:, 0].mean()
        if self.sqn:
            q = self.qhead(h).gather(1, pos.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                nxt = torch.cat([seq[:, 1:], (pos + 1).unsqueeze(1)], 1)
                q_next = self.q_target(self.net_target(nxt)).max(1).values
                target = rew + self.q_gamma * q_next
            loss = loss + self.q_weight * F.mse_loss(q, target)
        return loss

    def _train(self, data_t, max_epochs, patience, every, val_block, ctx, resume_key=None):
        u, seq, pos, rew = data_t
        seq_t = torch.as_tensor(seq, device=self.device)
        pos_t = torch.as_tensor(pos, device=self.device)
        rew_t = torch.as_tensor(rew, device=self.device, dtype=torch.float32)
        params = list(self.net.parameters()) + (list(self.qhead.parameters()) if self.sqn else [])
        opt = torch.optim.Adam(params, lr=self.lr)
        if resume_key and self.train_state and self.train_state.get("key") == resume_key:
            st = self.train_state
            opt.load_state_dict(st["opt"])
        else:
            st = {"key": resume_key, "epoch": 0, "best": -1.0, "bad": 0, "best_state": None}
        self._known = torch.as_tensor(np.where(self.seen_item)[0], device=self.device)
        n = len(pos)
        times = []
        self._training = True
        try:
            while st["epoch"] < max_epochs and n > 0:
                te = time.time()
                if self.sqn:
                    self.net_target = copy.deepcopy(self.net).eval()
                    self.q_target = copy.deepcopy(self.qhead).eval()
                self.net.train()
                perm = torch.randperm(n, device=self.device)
                for a in range(0, n, self.batch):
                    b = perm[a: a + self.batch]
                    loss = self._loss(seq_t[b], pos_t[b], rew_t[b])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                self.net.eval()
                st["epoch"] += 1
                times.append(time.time() - te)
                if val_block is not None and st["epoch"] % every == 0:
                    sc = ctx.val_metric(self, val_block)
                    if sc > st["best"] + 1e-5:
                        st["best"], st["bad"] = sc, 0
                        st["best_state"] = {"net": copy.deepcopy(self.net.state_dict()),
                                            "q": copy.deepcopy(self.qhead.state_dict()) if self.sqn else None}
                    else:
                        st["bad"] += 1
                    if st["bad"] >= patience:
                        break
                if resume_key and st["epoch"] % 5 == 0:
                    st["opt"] = opt.state_dict()
                    self.train_state = st
                    ctx.save()
                if not ctx.budget.ok(need_s=3 * float(np.mean(times)) + float(self.cfg.get("budget_margin_s", 120))):
                    if resume_key:
                        st["opt"] = opt.state_dict()
                        self.train_state = st
                        ctx.save()
                    raise TimeLimitReached("time budget reached during training")
        finally:
            self._training = False
            self.net.eval()
        if st["best_state"] is not None:
            self.net.load_state_dict(st["best_state"]["net"])
            if self.sqn and st["best_state"]["q"] is not None:
                self.qhead.load_state_dict(st["best_state"]["q"])
        self.train_state = None
        return {"epochs": st["epoch"], "best_val": st["best"], "epoch_s": float(np.mean(times)) if times else 0}

    def warmup(self, ctx):
        d = self.data
        W = d.warmup_blocks
        for b in W:
            _, i, _, _ = d.pairs(b, (PART_TRAIN, PART_VAL))
            self.seen_item[i] = True
        info = self._train(self._targets(W, W[-1]), self.ep_w_max, self.ep_w_pat, self.ep_w_every,
                           W[-1], ctx, resume_key="warmup")
        self.diag.update({"warmup_" + k: v for k, v in info.items()})
        return info

    def update(self, block, ctx):
        t0 = time.time()
        _, i, _, _ = self.data.pairs(block, (PART_TRAIN,))
        self.seen_item[i] = True
        info = self._train(self._targets([block], block), self.ep_s, self.ep_s_pat, 1, block, ctx)
        self.diag.update({"epochs": info["epochs"], "update_s": time.time() - t0})
        return info

    def finalize_block(self, block):
        _, i, _, _ = self.data.pairs(block, (PART_VAL,))
        self.seen_item[i] = True

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def prepare_eval(self, block, light=False):
        self.net.eval()
        seq = self._seq_eval(block)
        H = torch.zeros(self.nU, self.net.item.embedding_dim, device=self.device)
        for a in range(0, self.nU, 4096):
            H[a: a + 4096] = self.net(seq[a: a + 4096])
        self.cache = {"H": H, "E": self.net.item_emb().detach()}

    @torch.no_grad()
    def score(self, users):
        return self.cache["H"][users] @ self.cache["E"].T

    @torch.no_grad()
    def pair_mv(self, u, i):
        H, E = self.cache["H"], self.cache["E"]
        ut, it = torch.as_tensor(u, device=self.device), torch.as_tensor(i, device=self.device)
        return (H[ut] * E[it]).sum(-1).cpu().numpy(), None

    def state_dict(self):
        ts = self.train_state
        return {"net": self.net.state_dict(), "q": self.qhead.state_dict() if self.sqn else None,
                "seen_item": self.seen_item, "train_state": ts, "rng": self.rng.bit_generator.state,
                "diag": self.diag}

    def load_state_dict(self, st):
        self.net.load_state_dict(st["net"])
        if self.sqn and st["q"] is not None:
            self.qhead.load_state_dict(st["q"])
        self.seen_item = st["seen_item"]
        self.train_state = st["train_state"]
        if self.train_state and "opt" in self.train_state:
            from .graphrec import _opt_to
            self.train_state["opt"] = _opt_to(self.train_state["opt"], self.device)
        self.rng.bit_generator.state = st["rng"]
        self.diag = st.get("diag", {})

    def diagnostics(self):
        return dict(self.diag)
