"""GraphRec: TIDE-GR and every LightGCN-family baseline in one configurable class.

TIDE-GR (configs/method_tidegr.yaml) = all switches on:
  Gaussian belief (mu, lambda) + Laplace precision + drift-gated Bayesian forgetting,
  surprise-CUSUM drift detector, RL (contextual Thompson bandit) adaptation controller,
  sampled-softmax loss + XSimGCL noise contrast + recent-interest term,
  environment (trend) head with forecast, V-REx on common users,
  stratified replay memory with curvature hand-back, open-world priors,
  closed-form score variance + edge-sample variance, Thompson-sampling mode.
Baselines switch parts off (LightGCN-FT, LightGCN-retrain, XSimGCL-FT, EWC, KD-lite, PDA).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..drift import DriftDetector
from ..features import FeatureBank
from ..graph import GraphState, propagate, propagate_sq
from ..memory import ReplayMemory
from ..openworld import fit_text_map
from ..rl_bandit import DEFAULT_ARMS, LinTSBandit
from ..utils import TimeLimitReached, get_rng_state, log, set_rng_state
from .base import Method
from ..data.stream import PART_TRAIN, PART_VAL


class EnvHead(nn.Module):
    def __init__(self, p: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(p, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _g(cfg, path, default):
    v = cfg.get_path(path, default) if hasattr(cfg, "get_path") else default
    return default if v is None else v


class GraphRec(Method):
    supports_var = True
    supports_rel = True
    supports_ts = True

    def __init__(self, cfg, data, device, seed):
        super().__init__(cfg, data, device, seed)
        m = cfg.method
        self.mc = m
        self.nU, self.nI = data.n_users, data.n_items
        self.N = self.nU + self.nI
        self.d = int(m.get("dim", 64))
        self.L = int(m.get("layers", 2))
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.fb = FeatureBank(data)

        # switches
        self.anchor = m.get("anchor", "belief")  # belief | ewc | l2 | none
        self.belief = self.anchor in ("belief", "ewc")
        self.loss_kind = m.get("loss", "ssm")
        self.n_neg = int(m.get("n_neg", 64)) if self.loss_kind == "ssm" else 1
        self.tau_s = float(m.get("tau_s", 0.2))
        self.lambda_cl = float(_g(cfg, "method.contrast.lambda_cl", 0.0))
        self.cl_eps = float(_g(cfg, "method.contrast.eps", 0.2))
        self.cl_tau = float(_g(cfg, "method.contrast.tau", 0.2))
        self.cl_layer = int(_g(cfg, "method.contrast.layer", 1))
        self.gamma_r = float(_g(cfg, "method.recent.gamma", 0.0))
        self.n_recent = int(_g(cfg, "method.recent.n", 20))
        self.use_env = bool(m.get("env_head", True))
        self.pda = bool(m.get("pda", False))
        self.pda_gamma = float(m.get("pda_gamma", 0.1))
        self.lambda_inv = float(_g(cfg, "method.vrex.lambda_inv", 0.0))
        self.n_c = int(_g(cfg, "method.vrex.n_c", 1024))
        self.anchor_l2 = float(m.get("anchor_l2", 10.0))
        self.wd = float(m.get("weight_decay_reg", 1e-4))
        # prior precision in data units: lambda0 = n0 * (median curvature of one interaction),
        # set after warm-up; until then a provisional value is used.
        self.n0 = float(_g(cfg, "method.belief.n0", 1.0))
        self.lambda0 = float(_g(cfg, "method.belief.lambda0_init", 1.0))
        self.ts_scale = 1.0
        self.var_scale = 1.0  # = calibrated nu (set by the runner) for reliability-mode ranking
        self.gamma_F = float(_g(cfg, "method.belief.gamma_F", 1.0))
        self.q0 = float(_g(cfg, "method.belief.q0", 1e-4))
        self.q1 = float(_g(cfg, "method.belief.q1", 0.5))
        self.h = float(_g(cfg, "method.belief.h", 4.0))
        self.gate = bool(_g(cfg, "method.belief.gate", True))
        self.drift_on = bool(_g(cfg, "method.drift.enabled", self.anchor == "belief"))
        self.rl_on = bool(_g(cfg, "method.rl.enabled", False))
        self.fixed_arm = _g(cfg, "method.rl.fixed_arm", None)
        self.mem_on = bool(_g(cfg, "method.memory.enabled", False))
        self.ow_on = bool(_g(cfg, "method.openworld.enabled", False))
        self.zeta = float(_g(cfg, "method.openworld.zeta", 1.0))
        self.lambda_w = float(_g(cfg, "method.openworld.lambda_w", 1.0))
        self.rho0 = float(_g(cfg, "method.edge.rho0", 0.95))
        self.rhoinf = float(_g(cfg, "method.edge.rhoinf", 0.6))
        self.tau_rho = float(_g(cfg, "method.edge.tau_rho", 0.0))
        self.S = int(m.get("edge_samples", 0)) if self.belief else 0
        self.ts_on = bool(m.get("thompson", False)) and self.belief
        self.continual = m.get("continual", "ft")  # ft | retrain
        self.beta_r = float(m.get("beta_r", 0.5))
        self.lr = float(m.get("lr", 1e-3))
        self.batch = int(m.get("batch", 4096))
        ep = m.get("epochs", {}) or {}
        self.ep_w_max = int(ep.get("warmup_max", 200))
        self.ep_w_pat = int(ep.get("warmup_patience", 5))
        self.ep_w_every = int(ep.get("warmup_eval_every", 2))
        self.ep_s = int(ep.get("stream", 20))
        self.ep_s_pat = int(ep.get("stream_patience", 3))
        self.ep_s_every = int(ep.get("stream_eval_every", 1))
        self.alarm_mult = float(m.get("alarm_epoch_mult", 2.0))
        self.supports_var = self.belief
        self.supports_rel = self.belief
        self.supports_ts = self.ts_on

        dev = device
        self.graph = GraphState(self.nU, self.nI)
        self.theta = nn.Parameter(torch.randn(self.N, self.d, device=dev) * float(m.get("init_std", 0.1)))
        self.mu = self.theta.detach().clone()
        self.lam = torch.full((self.N, self.d), self.lambda0, device=dev)
        self.mu_pred = self.mu.clone()
        self.lam_pred = self.lam.clone()
        self.env = EnvHead(self.fb.p).to(dev)
        self.a_u = nn.Parameter(torch.zeros(self.nU, device=dev))
        self.phi_prev = None
        self.drift = DriftDetector(self.N, mode=_g(cfg, "method.drift.mode", "surprise"),
                                   delta=float(_g(cfg, "method.drift.delta", 0.5)),
                                   varsigma=float(_g(cfg, "method.drift.varsigma", 0.9)))
        arms = _g(cfg, "method.rl.arms", None) or DEFAULT_ARMS
        self.bandit = LinTSBandit(arms=arms, dim=8,
                                  prior_std=float(_g(cfg, "method.rl.prior_std", 0.01)),
                                  noise_std=float(_g(cfg, "method.rl.noise_std", 0.05)),
                                  delta0=float(_g(cfg, "method.rl.delta0", 0.02)), seed=seed)
        self.n_groups = int(_g(cfg, "method.rl.groups", 5))
        self.rl_random = bool(_g(cfg, "method.rl.random", False))
        self.rl_no_drift_ctx = bool(_g(cfg, "method.rl.no_drift_context", False))
        self.w_f = float(_g(cfg, "method.rl.w_f", 1.0))
        self.w_c = float(_g(cfg, "method.rl.w_c", 0.5))
        self.memory = ReplayMemory(frac=float(_g(cfg, "method.memory.frac", 0.02)),
                                   window=int(_g(cfg, "method.memory.window", 8)),
                                   season_slots=bool(_g(cfg, "method.memory.season_slots", True)))
        self.item_feat = None
        if data.item_feat is not None and self.ow_on:
            self.item_feat = torch.as_tensor(data.item_feat, device=dev)
        self.W = None
        self.var_res = None
        self.eta = 0.5
        self.last_ece = 0.0
        self.seen_user = np.zeros(self.nU, bool)
        self.seen_item = np.zeros(self.nI, bool)
        self.train_state = None  # resumable warm-up progress
        self.cache: dict = {}
        self._training = False
        self.diag: dict = {}

    # ================================================================== helpers
    def _w(self, block):
        return self.graph.weights(block, self.rho0, self.rhoinf, self.tau_rho)

    def _adj(self, block):
        return self.graph.adj(self._w(block), self.device, cache_key=("A", self.graph.version, block))

    def _alpha(self, users):
        return F.softplus(self.a_u[users])

    def _phi_params(self):
        return list(self.env.parameters()) + [self.a_u]

    def _x(self, arr):
        return torch.as_tensor(arr, device=self.device, dtype=torch.float32)

    def _pda_pop(self, counts: np.ndarray) -> torch.Tensor:
        c = torch.as_tensor(counts, device=self.device, dtype=torch.float32)
        return (c / c.max().clamp_min(1.0)).clamp_min(1e-6)

    def _pda_forecast(self, k):
        p1 = self.fb.counts(k - 1)
        p2 = self.fb.counts(k - 2)
        n1 = p1 / max(p1.max(), 1.0)
        n2 = p2 / max(p2.max(), 1.0)
        return torch.as_tensor(np.clip(n1 + (n1 - n2), 1e-6, None), device=self.device, dtype=torch.float32)

    def _recent_tensor(self, k):
        R = self.fb.recent(k, self.n_recent)
        return torch.as_tensor(R, device=self.device)

    @staticmethod
    def _recent_mean(R_rows: torch.Tensor, Zi: torch.Tensor):
        """mean of Zi over valid indices in R_rows [B,n] (-1 = pad); also count."""
        valid = R_rows >= 0
        idx = R_rows.clamp_min(0)
        E = Zi[idx] * valid.unsqueeze(-1)
        cnt = valid.sum(1, keepdim=True).clamp_min(1)
        return E.sum(1) / cnt, valid, cnt

    def _known_items(self) -> torch.Tensor:
        idx = np.where(self.seen_item)[0]
        if len(idx) == 0:
            idx = np.arange(self.nI)
        return torch.as_tensor(idx, device=self.device)

    # ================================================================== scoring core
    def _pair_scores(self, Zu, Zi_pos, Zi_neg, env_pos, env_neg, alpha, pop_pos=None, pop_neg=None):
        """Zu [B,d]; Zi_pos [B,d]; Zi_neg [B,n,d]; env_* env-head outputs [B],[B,n]."""
        sp = (Zu * Zi_pos).sum(-1)
        sn = torch.einsum("bd,bnd->bn", Zu, Zi_neg)
        if self.pda:
            sp = (F.elu(sp) + 1) * pop_pos.pow(self.pda_gamma)
            sn = (F.elu(sn) + 1) * pop_neg.pow(self.pda_gamma)
        elif self.use_env:
            sp = sp + alpha * env_pos
            sn = sn + alpha.unsqueeze(1) * env_neg
        return sp, sn

    def _rec_loss(self, sp, sn):
        if self.loss_kind == "bpr":
            return -F.logsigmoid(sp - sn[:, 0])
        logits = torch.cat([sp.unsqueeze(1), sn], 1) / self.tau_s
        return -F.log_softmax(logits, dim=1)[:, 0]

    @staticmethod
    def _infonce(a, b, tau):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = a @ b.T / tau
        return F.cross_entropy(logits, torch.arange(len(a), device=a.device))

    # ================================================================== training
    def _build_train(self, pairs_u, pairs_i, pairs_e, block_for_graph):
        """Tensors for one training call. pairs_e: environment (block id) per pair."""
        dev = self.device
        envs = sorted(set(int(e) for e in np.unique(pairs_e)))
        slot = {e: s for s, e in enumerate(envs)}
        X = torch.stack([self._x(self.fb.x_env(e)) for e in envs]) if self.use_env else None
        pops = torch.stack([self._pda_pop(self.fb.counts(e)) for e in envs]) if self.pda else None
        R = None
        if self.gamma_r > 0:
            R = torch.full((len(pairs_u), self.n_recent), -1, dtype=torch.long, device=dev)
            for e in envs:
                m = np.where(pairs_e == e)[0]
                Re = self.fb.recent(e, self.n_recent)[pairs_u[m]]
                R[torch.as_tensor(m, device=dev)] = torch.as_tensor(Re, device=dev)
        slots = torch.as_tensor(np.searchsorted(np.asarray(envs), pairs_e), device=dev)
        common = None
        if self.lambda_inv > 0 and len(envs) >= 2:
            ue = np.unique(np.stack([pairs_u, pairs_e]), axis=1)
            n_env_per_user = np.bincount(ue[0], minlength=self.nU)
            is_common = n_env_per_user[pairs_u] >= 2
            common = []
            for e in envs:
                idx = np.where((pairs_e == e) & is_common)[0]
                if len(idx) >= 32:
                    common.append(torch.as_tensor(idx, device=dev))
            if len(common) < 2:
                common = None
        seen_nodes = None
        if self.anchor == "l2" and self.phi_prev is not None:
            seen_nodes = torch.as_tensor(np.concatenate([np.where(self.seen_user)[0],
                                                         np.where(self.seen_item)[0] + self.nU]), device=dev)
        return dict(u=torch.as_tensor(pairs_u, device=dev), i=torch.as_tensor(pairs_i, device=dev),
                    slot=slots, X=X, pops=pops, R=R, common=common, n=len(pairs_u),
                    A=self._adj(block_for_graph)[0], known=self._known_items(), seen_nodes=seen_nodes)

    def _forward_full(self, A, noise: bool):
        return propagate(A, self.theta, self.L, noise_eps=self.cl_eps if noise else 0.0,
                         cl_layer=self.cl_layer)

    def _batch_loss(self, T, idx, Z, Zcl, N_k):
        dev = self.device
        u, i, sl = T["u"][idx], T["i"][idx], T["slot"][idx]
        Zu_all, Zi_all = Z[: self.nU], Z[self.nU:]
        Zu = Zu_all[u]
        if T["R"] is not None:
            rm, _, _ = self._recent_mean(T["R"][idx], Zi_all)
            Zu = Zu + self.gamma_r * rm
        known = T["known"]
        neg = known[torch.randint(len(known), (len(idx), self.n_neg), device=dev)]
        alpha = self._alpha(u)
        env_pos = env_neg = pop_pos = pop_neg = None
        if self.pda:
            pop_pos = T["pops"][sl, i]
            pop_neg = T["pops"][sl.unsqueeze(1), neg]
        elif self.use_env:
            hx = self.env(T["X"])  # [n_env, nI]
            env_pos = hx[sl, i]
            env_neg = hx[sl.unsqueeze(1), neg]
        sp, sn = self._pair_scores(Zu, Zi_all[i], Zi_all[neg], env_pos, env_neg, alpha, pop_pos, pop_neg)
        loss = self._rec_loss(sp, sn).mean()
        parts = {}
        # V-REx over environments on common users (equal-size samples)
        if T["common"] is not None and self.lambda_inv > 0:
            risks = []
            hx = self.env(T["X"]) if self.use_env and not self.pda else None
            for cidx in T["common"]:
                s = cidx[torch.randint(len(cidx), (min(self.n_c, len(cidx)),), device=dev)]
                uu, ii, ss = T["u"][s], T["i"][s], T["slot"][s]
                Zu2 = Zu_all[uu]
                if T["R"] is not None:
                    rm2, _, _ = self._recent_mean(T["R"][s], Zi_all)
                    Zu2 = Zu2 + self.gamma_r * rm2
                ng = known[torch.randint(len(known), (len(s), self.n_neg), device=dev)]
                ep = en = pp = pn = None
                if self.pda:
                    pp, pn = T["pops"][ss, ii], T["pops"][ss.unsqueeze(1), ng]
                elif hx is not None:
                    ep, en = hx[ss, ii], hx[ss.unsqueeze(1), ng]
                a2 = self._alpha(uu)
                sp2, sn2 = self._pair_scores(Zu2, Zi_all[ii], Zi_all[ng], ep, en, a2, pp, pn)
                risks.append(self._rec_loss(sp2, sn2).mean())
            R = torch.stack(risks)
            vrex = R.var(unbiased=False)
            loss = loss + self.lambda_inv * vrex
        # XSimGCL contrast
        if self.lambda_cl > 0 and Zcl is not None:
            uu = torch.unique(u)[:4096]
            ii = torch.unique(i)[:4096] + self.nU
            cl = self._infonce(Z[uu], Zcl[uu], self.cl_tau) + self._infonce(Z[ii], Zcl[ii], self.cl_tau)
            loss = loss + self.lambda_cl * cl
        # anchor / regularisation
        if self.anchor in ("belief", "ewc") and self.phi_prev is not None:
            an = 0.5 * (self.lam_pred * (self.theta - self.mu_pred) ** 2).sum() / N_k
            loss = loss + an
            if self.phi_prev is not None:
                ph = sum(((p - q) ** 2).sum() for p, q in zip(self._phi_params(), self.phi_prev))
                loss = loss + 0.5 * ph / N_k
        elif self.anchor == "l2" and T["seen_nodes"] is not None:
            sn_ = T["seen_nodes"]
            # uniform distillation-style anchor: weight x mean over previously seen nodes of ||theta - mu||^2
            an = 0.5 * self.anchor_l2 * ((self.theta[sn_] - self.mu_pred[sn_]) ** 2).sum(1).mean()
            loss = loss + an
            ego = torch.cat([self.theta[u], self.theta[i + self.nU]])
            loss = loss + self.wd * 0.5 * (ego ** 2).sum() / len(idx)
        else:
            ego = torch.cat([self.theta[u], self.theta[i + self.nU], self.theta[neg.reshape(-1) + self.nU]])
            loss = loss + self.wd * 0.5 * (ego ** 2).sum() / len(idx)
        if self.use_env and not self.pda:
            loss = loss + 1e-6 * (self.a_u ** 2).sum()
        return loss, parts

    def _train(self, T, max_epochs, patience, eval_every, val_block, ctx, resume_key=None):
        """Joint Adam training with early stopping on validation NDCG@20.
        If resume_key is set, progress is checkpointed via ctx.save() every few epochs."""
        params = [self.theta] + self._phi_params()
        opt = torch.optim.Adam(params, lr=self.lr)
        st = None
        if resume_key and self.train_state and self.train_state.get("key") == resume_key:
            st = self.train_state
            opt.load_state_dict(st["opt"])
            log(f"[train] resuming {resume_key} at epoch {st['epoch']}")
        else:
            st = {"key": resume_key, "epoch": 0, "best": -1.0, "bad": 0, "best_state": None}
        N_k = float(T["n"])
        n = T["n"]
        noise = self.lambda_cl > 0
        self._training = True
        t0 = time.time()
        epoch_times = []
        try:
            while st["epoch"] < max_epochs:
                te = time.time()
                perm = torch.randperm(n, device=self.device)
                for a in range(0, n, self.batch):
                    idx = perm[a: a + self.batch]
                    Z, Zcl = self._forward_full(T["A"], noise)
                    loss, _ = self._batch_loss(T, idx, Z, Zcl if noise else None, N_k)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                st["epoch"] += 1
                epoch_times.append(time.time() - te)
                if val_block is not None and (st["epoch"] % eval_every == 0 or st["epoch"] == max_epochs):
                    score = ctx.val_metric(self, val_block)
                    if self.cfg.get("debug"):
                        log(f"[train] epoch {st['epoch']} val_ndcg@20={score:.4f} ({time.time() - te:.1f}s)")
                    if score > st["best"] + 1e-5:
                        st["best"], st["bad"] = score, 0
                        st["best_state"] = self._snapshot()
                    else:
                        st["bad"] += 1
                    if st["bad"] >= patience:
                        break
                if resume_key and st["epoch"] % max(1, int(self.mc.get("ckpt_every_epochs", 5))) == 0:
                    st["opt"] = opt.state_dict()
                    self.train_state = st
                    ctx.save()
                if not ctx.budget.ok(need_s=3 * (np.mean(epoch_times) if epoch_times else 0) + float(self.cfg.get("budget_margin_s", 120))):
                    if resume_key:  # only warm-up is resumable mid-training; stream blocks restart
                        st["opt"] = opt.state_dict()
                        self.train_state = st
                        ctx.save()
                    raise TimeLimitReached("time budget reached during training")
        finally:
            self._training = False
        if st["best_state"] is not None:
            self._restore(st["best_state"])
        self.train_state = None
        info = {"epochs": st["epoch"], "best_val": st["best"], "train_s": time.time() - t0,
                "epoch_s": float(np.mean(epoch_times)) if epoch_times else 0.0}
        return info

    def _snapshot(self):
        return {"theta": self.theta.detach().clone(), "env": copy.deepcopy(self.env.state_dict()),
                "a_u": self.a_u.detach().clone()}

    def _restore(self, s):
        with torch.no_grad():
            self.theta.copy_(s["theta"])
            self.a_u.copy_(s["a_u"])
        self.env.load_state_dict(s["env"])

    # ================================================================== Laplace
    @torch.no_grad()
    def _curvature(self, u: np.ndarray, i: np.ndarray, block: int) -> torch.Tensor:
        """Diagonal GGN of the ranking loss w.r.t. propagated reps, pushed back with P^{o2}
        (v1 Eq. 16-17; softmax version in TIDE-GR_FINAL §4.4)."""
        dev = self.device
        G = torch.zeros(self.N, self.d, device=dev)
        if len(u) == 0:
            return G
        A, A2 = self._adj(block)
        Z, _ = propagate(A, self.mu, self.L)
        Zu_all, Zi_all = Z[: self.nU], Z[self.nU:]
        known = self._known_items()
        Rk = self._recent_tensor(block) if self.gamma_r > 0 else None
        ut, it = torch.as_tensor(u, device=dev), torch.as_tensor(i, device=dev)
        for a in range(0, len(u), self.batch):
            uu, ii = ut[a: a + self.batch], it[a: a + self.batch]
            Zu = Zu_all[uu]
            if Rk is not None:
                rm, _, _ = self._recent_mean(Rk[uu], Zi_all)
                Zu = Zu + self.gamma_r * rm
            neg = known[torch.randint(len(known), (len(uu), self.n_neg), device=dev)]
            Zp, Zn = Zi_all[ii], Zi_all[neg]
            if self.loss_kind == "bpr":
                x = (Zu * Zp).sum(-1) - (Zu * Zn[:, 0]).sum(-1)
                c = torch.sigmoid(x) * (1 - torch.sigmoid(x))
                G.index_add_(0, uu, c.unsqueeze(1) * (Zp - Zn[:, 0]) ** 2)
                cz = c.unsqueeze(1) * Zu ** 2
                G.index_add_(0, ii + self.nU, cz)
                G.index_add_(0, neg[:, 0] + self.nU, cz)
            else:
                cand = torch.cat([Zp.unsqueeze(1), Zn], 1)  # [B,1+n,d]
                logits = torch.einsum("bd,bnd->bn", Zu, cand) / self.tau_s
                p = torch.softmax(logits, 1)
                c = p * (1 - p) / self.tau_s ** 2  # [B,1+n]
                G.index_add_(0, uu, (c.unsqueeze(-1) * cand ** 2).sum(1))
                items = torch.cat([ii.unsqueeze(1), neg], 1)
                cz = c.unsqueeze(-1) * (Zu ** 2).unsqueeze(1)
                G.index_add_(0, (items + self.nU).reshape(-1), cz.reshape(-1, self.d))
        return propagate_sq(A2, G, self.L)

    # ================================================================== belief dynamics
    @torch.no_grad()
    def _predict_step(self, active_nodes: np.ndarray, q1_node: np.ndarray):
        if self.anchor != "belief":
            self.mu_pred = self.mu.clone()
            self.lam_pred = self.lam.clone()
            return
        D = torch.as_tensor(self.drift.D, device=self.device, dtype=torch.float32)
        q = torch.full((self.N,), self.q0, device=self.device)
        act = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        act[torch.as_tensor(active_nodes, device=self.device)] = True
        q1 = torch.as_tensor(q1_node, device=self.device, dtype=torch.float32)
        if self.gate and self.drift_on:
            q = q + act * q1 * torch.tanh(D / self.h)
        else:
            q = q + act * q1 * 0.5
        q = (q / self.lambda0).unsqueeze(1)
        self.lam_pred = 1.0 / (1.0 / self.lam + q)
        self.mu_pred = self.mu.clone()

    @torch.no_grad()
    def _admit(self, block: int):
        """Open-world priors for users/items first seen in this block."""
        d = self.data
        new_items = np.where(d.item_first_block == block)[0]
        new_users = np.where(d.user_first_block == block)[0]
        if len(new_items):
            ni = torch.as_tensor(new_items, device=self.device)
            known = np.where(self.seen_item)[0]
            if self.ow_on and self.W is not None and self.item_feat is not None:
                mu0 = self.item_feat[ni] @ self.W
                lam0 = (1.0 / self.var_res).clamp(max=self.lambda0).unsqueeze(0).expand(len(ni), -1)
            elif self.ow_on and len(known):
                mu0 = self.mu[self.nU + torch.as_tensor(known, device=self.device)].mean(0, keepdim=True).expand(len(ni), -1)
                lam0 = torch.full((len(ni), self.d), self.lambda0, device=self.device)
            else:
                mu0, lam0 = self.mu[self.nU + ni], torch.full((len(ni), self.d), self.lambda0, device=self.device)
            self.mu[self.nU + ni] = mu0
            self.lam[self.nU + ni] = lam0
        if len(new_users) and self.ow_on:
            u, i, _, _ = d.pairs(block, (PART_TRAIN,))
            sel = np.isin(u, new_users)
            A, _ = self._adj(block)
            Z, _ = propagate(A, self.mu, self.L)
            Zi = Z[self.nU:].clone()
            is_new_item = torch.zeros(self.nI, dtype=torch.bool, device=self.device)
            if len(new_items):
                is_new_item[torch.as_tensor(new_items, device=self.device)] = True
                Zi[is_new_item] = self.mu[self.nU:][is_new_item]
            su, si = torch.as_tensor(u[sel], device=self.device), torch.as_tensor(i[sel], device=self.device)
            acc = torch.zeros(self.nU, self.d, device=self.device)
            cnt = torch.zeros(self.nU, device=self.device)
            acc.index_add_(0, su, Zi[si])
            cnt.index_add_(0, su, torch.ones_like(su, dtype=torch.float32))
            nu = torch.as_tensor(new_users, device=self.device)
            has = cnt[nu] > 0
            seen_u = np.where(self.seen_user)[0]
            fallback = self.mu[torch.as_tensor(seen_u, device=self.device)].mean(0) if len(seen_u) else self.mu[nu].mean(0)
            mu0 = torch.where(has.unsqueeze(1), self.zeta * acc[nu] / cnt[nu].clamp_min(1).unsqueeze(1),
                              fallback.unsqueeze(0).expand(len(nu), -1))
            self.mu[nu] = mu0
            self.lam[nu] = self.lambda0
        with torch.no_grad():
            if len(new_items) or len(new_users):
                nodes = np.concatenate([new_users, new_items + self.nU])
                nt = torch.as_tensor(nodes, device=self.device)
                self.theta[nt] = self.mu[nt]

    @torch.no_grad()
    def _fit_openworld(self):
        if not (self.ow_on and self.item_feat is not None):
            return
        warm = np.where(self.seen_item & (self.graph.degree_items() >= 2))[0]
        if len(warm) < 20:
            return
        wi = torch.as_tensor(warm, device=self.device)
        W, var = fit_text_map(self.item_feat[wi], self.mu[self.nU + wi], self.lam[self.nU + wi], self.lambda_w)
        if W is not None:
            self.W, self.var_res = W, var

    def _mark_seen(self, block: int, parts=(PART_TRAIN,)):
        u, i, _, _ = self.data.pairs(block, parts)
        self.seen_user[u] = True
        self.seen_item[i] = True

    # ================================================================== warm-up
    def warmup(self, ctx):
        d = self.data
        W = d.warmup_blocks
        last = W[-1]
        if self.train_state is None or self.train_state.get("key") != "warmup":
            # build graph: train(+val except last block) edges of all warm-up blocks
            self.graph = GraphState(self.nU, self.nI)
            for b in W:
                u, i, _, _ = d.pairs(b, (PART_TRAIN,) if b == last else (PART_TRAIN, PART_VAL))
                self.graph.add(u, i, b)
                self._mark_seen(b, (PART_TRAIN,) if b == last else (PART_TRAIN, PART_VAL))
        us, is_, es = [], [], []
        for b in W:
            u, i, _, _ = d.pairs(b, (PART_TRAIN,))
            us.append(u), is_.append(i), es.append(np.full(len(u), b))
        pu, pi, pe = np.concatenate(us), np.concatenate(is_), np.concatenate(es)
        T = self._build_train(pu, pi, pe, last)
        self.mu_pred = torch.zeros_like(self.mu)  # prior N(0, 1/lambda0)
        self.lam_pred = torch.full_like(self.lam, self.lambda0)
        info = self._train(T, self.ep_w_max, self.ep_w_pat, self.ep_w_every, last, ctx, resume_key="warmup")
        with torch.no_grad():
            self.mu = self.theta.detach().clone()
        # memory: fill from warm-up blocks, then Laplace on everything not held in memory
        if self.mem_on:
            for b in W:
                u, i, _, _ = d.pairs(b, (PART_TRAIN,))
                sel = self.memory.select(u, i, np.ones(len(u)), self.rng)
                vu, vi, _, _ = d.pairs(b, (PART_VAL,))
                self.memory.insert(b, int(d.block_season[b]), u[sel], i[sel], vu, vi, self.rng)
        if self.belief:
            mu_, mi_, _ = self.memory.all_pairs() if self.mem_on else (np.zeros(0, int),) * 3
            keep = np.ones(len(pu), bool)
            if len(mu_):
                inmem = np.isin(pu * self.nI + pi, mu_ * self.nI + mi_)
                keep &= ~inmem
            curv = self._curvature(pu[keep], pi[keep], last)
            # median curvature contributed by one interaction (per node, averaged over dims)
            deg = np.bincount(np.concatenate([pu[keep], pi[keep] + self.nU]), minlength=self.N)
            has = torch.as_tensor(deg > 0, device=self.device)
            per = curv.mean(1)[has] / torch.as_tensor(deg[deg > 0], device=self.device, dtype=torch.float32)
            unit = float(per.median()) if len(per) else 1.0
            self.lambda0 = max(self.n0 * unit, 1e-8)
            self.lam = torch.full_like(self.lam, self.lambda0) + self.gamma_F * curv
            self.diag["lambda_unit_per_interaction"] = unit
            # Thompson sampling scale: posterior std of well-evidenced nodes ~ half the typical |mu|
            seen = torch.as_tensor(np.concatenate([self.seen_user, self.seen_item]), device=self.device)
            sd = (1.0 / torch.sqrt(self.lam[seen])).median()
            self.ts_scale = float(0.5 * self.mu[seen].abs().median() / sd.clamp_min(1e-12))
        self.phi_prev = [p.detach().clone() for p in self._phi_params()]
        for b in W[1:]:
            self.drift.js_warm.append(self._js(b))
        self.eta = self.fb.fit_eta(last + 1)
        self._fit_openworld()
        self.diag.update({"warmup_" + k: v for k, v in info.items()})
        self._log_lambda("warmup")
        return info

    def _js(self, b):
        from ..drift import js_divergence
        return js_divergence(self.fb.counts(b - 1), self.fb.counts(b))

    def _log_lambda(self, tag):
        if not self.belief:
            return
        lam_u = self.lam[: self.nU].mean(1)
        deg = torch.as_tensor(self.graph.degree_users(), device=self.device)
        lo = lam_u[(deg > 0) & (deg <= 3)]
        hi = lam_u[deg >= 20]
        self.diag["lambda_user_lowdeg_median"] = float(lo.median()) if len(lo) else None
        self.diag["lambda_user_highdeg_median"] = float(hi.median()) if len(hi) else None
        log(f"[belief:{tag}] median lambda low-degree users={self.diag['lambda_user_lowdeg_median']} "
            f"high-degree users={self.diag['lambda_user_highdeg_median']} (lambda0={self.lambda0:.3g})")

    # ================================================================== stream protocol
    def observe_test(self, block: int, info: dict):
        """Called after the runner tested block `block` (before training on it)."""
        self.last_ece = float(info.get("ece", 0.0))
        if self.rl_on and self.bandit.pending:
            pu, py, pp = info["pair_u"], info["pair_y"], info["pair_p"]

            def ece_fn(users):
                from ..eval.metrics import ece_score
                m = np.isin(pu, np.asarray(users))
                return ece_score(pp[m], py[m]) if m.sum() > 50 else self.last_ece

            ndcg = dict(zip(info["users"].tolist(), info["ndcg"].tolist()))
            self.bandit.reward_from(ndcg, ece_fn, self.w_f, self.w_c)
        if self.drift_on:
            pos = info["pair_y"] == 1
            active = np.unique(np.concatenate([info["pair_u"][pos], info["pair_i"][pos] + self.nU]))
            self.drift.update(info["pair_u"], info["pair_i"] + self.nU, info["pair_y"], info["pair_p"],
                              active, self.fb.counts(block - 1), self.fb.counts(block))

    def _rl_decide(self, block, u_blk, i_blk):
        """Choose arms per drift group; returns (q1 per node, epoch_mult, replay_ratio, arm names)."""
        arms = self.bandit.arms
        users = np.unique(u_blk)
        q1_node = np.full(self.N, self.q1, np.float32)
        if not self.rl_on:
            a = int(self.fixed_arm) if self.fixed_arm is not None else None
            if a is None:
                return q1_node, 1.0, 1.0, ["fixed_default"]
            q1_node[:] = arms[a]["q1"]
            return q1_node, arms[a]["epoch_mult"], arms[a]["replay_ratio"], [arms[a]["name"]]
        D = self.drift.D
        order = users[np.argsort(D[users], kind="stable")]
        groups = [g for g in np.array_split(order, self.n_groups) if len(g)]
        new_u = float(np.mean(self.data.user_first_block[users] == block)) if len(users) else 0.0
        items = np.unique(i_blk)
        new_i = float(np.mean(self.data.item_first_block[items] == block)) if len(items) else 0.0
        xs, chosen = [], []
        for g in groups:
            x = np.array([
                np.clip(self.drift.last_S[g].mean(), -10, 10) / 5.0,
                float(np.mean(D[g] > self.h)),
                min(self.drift.last_js * 10.0, 5.0),
                min(self.last_ece * 10.0, 5.0),
                new_u, new_i,
                np.log1p(len(u_blk)) / 15.0,
                1.0,
            ])
            if self.rl_no_drift_ctx:
                x[0:3] = 0.0  # ablation A11: controller without drift/uncertainty signals
            a = int(self.rng.integers(len(arms))) if self.rl_random else self.bandit.choose(x)
            xs.append(x.tolist())
            chosen.append(a)
            q1_node[g] = arms[a]["q1"]
        vals, cnts = np.unique(chosen, return_counts=True)
        blk_arm = int(vals[np.argmax(cnts)])
        q1_node[self.nU:] = arms[blk_arm]["q1"]
        self.bandit.record(block, [g.tolist() for g in groups], xs, chosen)
        return q1_node, arms[blk_arm]["epoch_mult"], arms[blk_arm]["replay_ratio"], [arms[a]["name"] for a in chosen]

    def update(self, block: int, ctx):
        d = self.data
        t0 = time.time()
        u, i, _, _ = d.pairs(block, (PART_TRAIN,))
        q1_node, ep_mult, replay_ratio, arm_names = self._rl_decide(block, u, i)
        if (not self.rl_on) and self.drift_on and self.drift.alarm:
            ep_mult *= self.alarm_mult
        active = np.unique(np.concatenate([u, i + self.nU]))
        forget_before = ctx.memory_ndcg(self, block) if (self.rl_on and self.mem_on) else None

        if self.continual == "retrain":
            return self._retrain(block, ctx, t0)

        self._admit(block)
        self._predict_step(active, q1_node)
        self.graph.add(u, i, block)
        self._mark_seen(block, (PART_TRAIN,))
        epochs = int(round(self.ep_s * ep_mult))
        info = {"epochs": 0}
        if epochs > 0 and len(u):
            n_rep = int(round(replay_ratio * len(u))) if self.mem_on else 0
            mu_, mi_, me_ = self.memory.sample(n_rep, self.rng) if n_rep else (np.zeros(0, int),) * 3
            pu = np.concatenate([u, mu_])
            pi = np.concatenate([i, mi_])
            pe = np.concatenate([np.full(len(u), block), me_])
            T = self._build_train(pu, pi, pe, block)
            with torch.no_grad():
                self.theta.copy_(self.mu)
            info = self._train(T, epochs, self.ep_s_pat, self.ep_s_every, block, ctx)
        with torch.no_grad():
            self.mu = self.theta.detach().clone()
        # memory selection, Laplace (memory edges excluded), eviction hand-back
        sel = np.zeros(0, int)
        if self.mem_on:
            D = self.drift.D
            prio = self.rho0 * np.exp(-D[u] / self.h) * np.exp(-D[i + self.nU] / self.h)
            sel = self.memory.select(u, i, prio, self.rng)
        if self.belief:
            keep = np.ones(len(u), bool)
            keep[sel] = False
            self.lam = self.lam_pred + self.gamma_F * self._curvature(u[keep], i[keep], block)
        if self.mem_on:
            vu, vi, _, _ = d.pairs(block, (PART_VAL,))
            ev_u, ev_i = self.memory.insert(block, int(d.block_season[block]), u[sel], i[sel], vu, vi, self.rng)
            if self.belief and len(ev_u):
                self.lam = self.lam + self.gamma_F * self._curvature(ev_u, ev_i, block)
        self.phi_prev = [p.detach().clone() for p in self._phi_params()]
        self.eta = self.fb.fit_eta(block + 1)
        self._fit_openworld()
        if forget_before is not None:
            after = ctx.memory_ndcg(self, block)
            self.bandit.set_forgetting(forget_before - after)
        self.diag.update({"arms": arm_names, "epochs": info.get("epochs"), "alarm": bool(self.drift.alarm),
                          "update_s": time.time() - t0, "memory_size": self.memory.size(),
                          "eta": self.eta, "mean_D_users": float(self.drift.D[: self.nU][self.seen_user].mean())
                          if self.seen_user.any() else 0.0})
        self._log_lambda(f"block{block}")
        return info

    def _retrain(self, block, ctx, t0):
        """LightGCN-retrain baseline: re-initialise and train on all history."""
        d = self.data
        u, i, _, _ = d.pairs(block, (PART_TRAIN,))
        self.graph.add(u, i, block)
        self._mark_seen(block, (PART_TRAIN,))
        with torch.no_grad():
            self.theta.copy_(torch.randn_like(self.theta) * float(self.mc.get("init_std", 0.1)))
        self.env = EnvHead(self.fb.p).to(self.device)
        with torch.no_grad():
            self.a_u.zero_()
        us, is_, es = [], [], []
        for b in range(block + 1):
            uu, ii, _, _ = d.pairs(b, (PART_TRAIN,))
            us.append(uu), is_.append(ii), es.append(np.full(len(uu), b))
        T = self._build_train(np.concatenate(us), np.concatenate(is_), np.concatenate(es), block)
        info = self._train(T, self.ep_w_max, self.ep_w_pat, self.ep_w_every, block, ctx)
        with torch.no_grad():
            self.mu = self.theta.detach().clone()
        self.eta = self.fb.fit_eta(block + 1)
        self.diag.update({"epochs": info.get("epochs"), "update_s": time.time() - t0})
        return info

    def finalize_block(self, block: int):
        u, i, _, _ = self.data.pairs(block, (PART_VAL,))
        self.graph.add(u, i, block)
        self._mark_seen(block, (PART_VAL,))

    # ================================================================== prediction
    @torch.no_grad()
    def prepare_eval(self, block: int, light: bool = False):
        P = self.theta.detach() if self._training else self.mu
        A, A2 = self._adj(block)
        m, _ = propagate(A, P, self.L)
        c = {"block": block, "m_i": m[self.nU:], "light": light}
        mu_u = m[: self.nU]
        Rk = self._recent_tensor(block) if self.gamma_r > 0 else None
        if Rk is not None:
            rm = torch.zeros_like(mu_u)
            for a in range(0, self.nU, 8192):
                rm[a: a + 8192], _, _ = self._recent_mean(Rk[a: a + 8192], c["m_i"])
            mu_u = mu_u + self.gamma_r * rm
        c["m_u"] = mu_u
        if self.pda:
            c["pop_hat"] = self._pda_forecast(block).pow(self.pda_gamma)
        elif self.use_env:
            c["h_i"] = self.env(self._x(self.fb.x_forecast(block, self.eta)))
            c["alpha"] = F.softplus(self.a_u)
        if self.belief and not light:
            s = propagate_sq(A2, 1.0 / self.lam, self.L)
            s_u, s_i = s[: self.nU], s[self.nU:]
            if Rk is not None:
                valid = Rk >= 0
                cnt = valid.sum(1, keepdim=True).clamp_min(1).float()
                add = torch.zeros_like(s_u)
                for a in range(0, self.nU, 8192):
                    r = Rk[a: a + 8192]
                    v = r >= 0
                    add[a: a + 8192] = (s_i[r.clamp_min(0)] * v.unsqueeze(-1)).sum(1) / cnt[a: a + 8192] ** 2
                s_u = s_u + self.gamma_r ** 2 * add
            c["s_u"], c["s_i"] = s_u, s_i
            c["edge_Z"] = []
            if self.S > 0:
                w = self._w(block)
                for _ in range(self.S):
                    keep = (self.rng.random(len(w)) < w).astype(np.float32)
                    As, _ = self.graph.adj(keep, self.device, cache_key=None)
                    Zs, _ = propagate(As, P, self.L)
                    zu = Zs[: self.nU]
                    if Rk is not None:
                        rm = torch.zeros_like(zu)
                        for a in range(0, self.nU, 8192):
                            rm[a: a + 8192], _, _ = self._recent_mean(Rk[a: a + 8192], Zs[self.nU:])
                        zu = zu + self.gamma_r * rm
                    c["edge_Z"].append((zu, Zs[self.nU:]))
            if self.ts_on:
                th = P + self.ts_scale * torch.randn_like(P) / torch.sqrt(self.lam)
                Zt, _ = propagate(A, th, self.L)
                zu = Zt[: self.nU]
                if Rk is not None:
                    rm = torch.zeros_like(zu)
                    for a in range(0, self.nU, 8192):
                        rm[a: a + 8192], _, _ = self._recent_mean(Rk[a: a + 8192], Zt[self.nU:])
                    zu = zu + self.gamma_r * rm
                c["ts_u"], c["ts_i"] = zu, Zt[self.nU:]
        self.cache = c

    def _env_term(self, users):
        c = self.cache
        if "h_i" in c:
            return c["alpha"][users].unsqueeze(1) * c["h_i"].unsqueeze(0)
        return 0.0

    @torch.no_grad()
    def score(self, users):
        c = self.cache
        s = c["m_u"][users] @ c["m_i"].T
        if self.pda:
            return (F.elu(s) + 1) * c["pop_hat"].unsqueeze(0)
        return s + self._env_term(users)

    @torch.no_grad()
    def _var_matrix(self, users):
        c = self.cache
        mu_, mi = c["m_u"][users], c["m_i"]
        su, si = c["s_u"][users], c["s_i"]
        v = (mu_ ** 2) @ si.T + su @ (mi ** 2).T + su @ si.T
        if c["edge_Z"]:  # running variance over edge samples (memory: 2 matrices, not S)
            n, mean, m2 = 0, None, None
            for zu, zi in c["edge_Z"]:
                x = zu[users] @ zi.T
                n += 1
                if mean is None:
                    mean, m2 = x, torch.zeros_like(x)
                else:
                    delta = x - mean
                    mean = mean + delta / n
                    m2 = m2 + delta * (x - mean)
            if n > 1:
                v = v + m2 / (n - 1)
        return v

    @torch.no_grad()
    def score_rel(self, users):
        if not self.belief:
            return self.score(users)
        return self.score(users) - self.beta_r * torch.sqrt(self.var_scale * self._var_matrix(users).clamp_min(0))

    @torch.no_grad()
    def score_ts(self, users):
        c = self.cache
        if "ts_u" not in c:
            return self.score(users)
        return c["ts_u"][users] @ c["ts_i"].T + self._env_term(users)

    @torch.no_grad()
    def pair_mv(self, u: np.ndarray, i: np.ndarray):
        c = self.cache
        ut = torch.as_tensor(u, device=self.device)
        it = torch.as_tensor(i, device=self.device)
        mu_, mi = c["m_u"][ut], c["m_i"][it]
        m = (mu_ * mi).sum(-1)
        if self.pda:
            m = (F.elu(m) + 1) * c["pop_hat"][it]
        elif "h_i" in c:
            m = m + c["alpha"][ut] * c["h_i"][it]
        if not (self.belief and "s_u" in c):
            return m.cpu().numpy(), None
        su, si = c["s_u"][ut], c["s_i"][it]
        v = (mu_ ** 2 * si + mi ** 2 * su + su * si).sum(-1)
        if c["edge_Z"]:
            S = torch.stack([(zu[ut] * zi[it]).sum(-1) for zu, zi in c["edge_Z"]])
            v = v + S.var(0, unbiased=True)
        return m.cpu().numpy(), v.cpu().numpy()

    # ================================================================== state
    def state_dict(self) -> dict:
        ts = None
        if self.train_state is not None:
            ts = dict(self.train_state)
        return {
            "theta": self.theta.detach().cpu(), "mu": self.mu.cpu(), "lam": self.lam.cpu(),
            "mu_pred": self.mu_pred.cpu(), "lam_pred": self.lam_pred.cpu(),
            "env": self.env.state_dict(), "a_u": self.a_u.detach().cpu(),
            "phi_prev": None if self.phi_prev is None else [p.cpu() for p in self.phi_prev],
            "graph": self.graph.state_dict(), "drift": self.drift.state_dict(),
            "bandit": self.bandit.state_dict(), "memory": self.memory.state_dict(),
            "W": None if self.W is None else self.W.cpu(),
            "var_res": None if self.var_res is None else self.var_res.cpu(),
            "eta": self.eta, "last_ece": self.last_ece, "lambda0": self.lambda0,
            "ts_scale": self.ts_scale, "var_scale": self.var_scale,
            "seen_user": self.seen_user, "seen_item": self.seen_item,
            "train_state": ts, "rng": self.rng.bit_generator.state, "diag": self.diag,
        }

    def load_state_dict(self, st: dict):
        dev = self.device
        with torch.no_grad():
            self.theta.copy_(st["theta"].to(dev))
            self.a_u.copy_(st["a_u"].to(dev))
        self.mu, self.lam = st["mu"].to(dev), st["lam"].to(dev)
        self.mu_pred, self.lam_pred = st["mu_pred"].to(dev), st["lam_pred"].to(dev)
        self.env.load_state_dict(st["env"])
        self.phi_prev = None if st["phi_prev"] is None else [p.to(dev) for p in st["phi_prev"]]
        self.graph.load_state_dict(st["graph"])
        self.drift.load_state_dict(st["drift"])
        self.bandit.load_state_dict(st["bandit"])
        self.memory.load_state_dict(st["memory"])
        self.W = None if st["W"] is None else st["W"].to(dev)
        self.var_res = None if st["var_res"] is None else st["var_res"].to(dev)
        self.eta, self.last_ece = st["eta"], st["last_ece"]
        self.lambda0 = st.get("lambda0", self.lambda0)
        self.ts_scale, self.var_scale = st.get("ts_scale", 1.0), st.get("var_scale", 1.0)
        self.seen_user, self.seen_item = st["seen_user"], st["seen_item"]
        ts = st.get("train_state")
        if ts is not None:
            ts = dict(ts)
            if ts.get("best_state") is not None:
                ts["best_state"] = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ts["best_state"].items()}
            if "opt" in ts:
                ts["opt"] = _opt_to(ts["opt"], dev)
        self.train_state = ts
        self.rng.bit_generator.state = st["rng"]
        self.diag = st.get("diag", {})

    def diagnostics(self) -> dict:
        return dict(self.diag)


def _opt_to(opt_state, dev):
    out = copy.deepcopy(opt_state)
    for s in out.get("state", {}).values():
        for k, v in s.items():
            if torch.is_tensor(v) and k != "step":
                s[k] = v.to(dev)
    return out
