"""3-parameter probit calibration (v1 Eq. 23) and delayed-feedback ACI (v1 Eq. 35).

p = sigmoid( (kappa*m + b) / sqrt(1 + pi*nu*kappa^2*v/8) ),  kappa>0, nu>=0.
nu = 0 (or use_var=False) is Platt scaling.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


LOG_KAPPA_MAX = 30.0  # |log kappa| bound: a degenerate LBFGS fit must never overflow exp()


class Calibrator:
    def __init__(self, use_var: bool = True):
        self.use_var = bool(use_var)
        self.log_kappa = 0.0
        self.b = -4.6  # ~ logit(0.01): base rate with 99 negatives per positive
        self.raw_nu = -2.0
        self.fitted = False

    @property
    def kappa(self):
        return math.exp(min(max(self.log_kappa, -LOG_KAPPA_MAX), LOG_KAPPA_MAX))

    @property
    def nu(self):
        return float(F.softplus(torch.tensor(self.raw_nu))) if self.use_var else 0.0

    def _logits(self, m, v, lk, b, rn):
        kappa = torch.exp(lk.clamp(-LOG_KAPPA_MAX, LOG_KAPPA_MAX))
        z = kappa * m + b
        if self.use_var and v is not None:
            nu = F.softplus(rn)
            z = z / torch.sqrt(1.0 + math.pi * nu * kappa ** 2 * v / 8.0)
        return z

    def fit(self, m: np.ndarray, v: np.ndarray | None, y: np.ndarray, iters: int = 100):
        if len(y) < 20 or y.min() == y.max():
            return self
        dev = "cpu"
        mt = torch.as_tensor(m, dtype=torch.float64, device=dev)
        vt = None if v is None else torch.as_tensor(v, dtype=torch.float64, device=dev).clamp_min(0)
        yt = torch.as_tensor(y, dtype=torch.float64, device=dev)
        # standardise for a well-conditioned start
        s = mt.std().clamp_min(1e-6)
        lk = torch.tensor(float(-torch.log(s)), dtype=torch.float64, requires_grad=True)
        b = torch.tensor(float(math.log(max(y.mean(), 1e-4) / max(1 - y.mean(), 1e-4)) - float(mt.mean() / s)),
                         dtype=torch.float64, requires_grad=True)
        rn = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
        params = [lk, b, rn] if (self.use_var and vt is not None) else [lk, b]
        opt = torch.optim.LBFGS(params, lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = self._logits(mt, vt, lk, b, rn)
            loss = F.binary_cross_entropy_with_logits(z, yt)
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except Exception:
            return self
        # keep the previous fit if LBFGS diverged (non-finite or an absurd scale, seen with nearly
        # constant scores, e.g. the no-boosters ablation trained with plain BPR)
        if all(torch.isfinite(p).all() for p in params) and abs(float(lk)) < LOG_KAPPA_MAX:
            self.log_kappa, self.b, self.raw_nu = float(lk), float(b), float(rn)
            self.fitted = True
        return self

    def predict(self, m, v=None) -> np.ndarray:
        mt = torch.as_tensor(np.asarray(m), dtype=torch.float64)
        vt = None if v is None else torch.as_tensor(np.asarray(v), dtype=torch.float64).clamp_min(0)
        z = self._logits(mt, vt, torch.tensor(self.log_kappa, dtype=torch.float64),
                         torch.tensor(self.b, dtype=torch.float64),
                         torch.tensor(self.raw_nu, dtype=torch.float64))
        return torch.sigmoid(z).numpy()

    def predict_torch(self, m: torch.Tensor, v: torch.Tensor | None = None) -> torch.Tensor:
        dt = m.dtype
        z = self._logits(m.double(), None if v is None else v.double().clamp_min(0),
                         torch.tensor(self.log_kappa, dtype=torch.float64, device=m.device),
                         torch.tensor(self.b, dtype=torch.float64, device=m.device),
                         torch.tensor(self.raw_nu, dtype=torch.float64, device=m.device))
        return torch.sigmoid(z).to(dt)

    def epistemic(self, m, v) -> np.ndarray:
        """EU ~ (sigma'(kappa m + b))^2 kappa^2 nu v   (v1 Eq. 24)."""
        z = np.clip(self.kappa * np.asarray(m, dtype=np.float64) + self.b, -60.0, 60.0)
        s = 1 / (1 + np.exp(-z))
        return (s * (1 - s)) ** 2 * self.kappa ** 2 * self.nu * np.asarray(v)

    def state_dict(self):
        return dict(log_kappa=self.log_kappa, b=self.b, raw_nu=self.raw_nu,
                    use_var=self.use_var, fitted=self.fitted)

    def load_state_dict(self, d):
        self.__dict__.update(d)


class ACI:
    """Adaptive conformal-style threshold on the selective error, with delay d."""

    def __init__(self, alpha: float = 0.5, gamma: float = 0.05, delay: int = 1,
                 tau0: float = 0.0, tau_min: float = 0.0, tau_max: float = 1.0):
        self.alpha, self.gamma, self.delay = float(alpha), float(gamma), int(delay)
        self.tau, self.tau_min, self.tau_max = float(tau0), float(tau_min), float(tau_max)
        self.pending: deque = deque()
        self.t = 0
        self.err_sum = 0.0
        self.cov_sum = 0.0

    def run_block(self, round_id: np.ndarray, conf: np.ndarray, err: np.ndarray):
        """Process users of one block, round by round. Returns served mask."""
        served = np.zeros(len(conf), bool)
        if len(conf) == 0:
            return served
        for r in np.unique(round_id):
            m = round_id == r
            s = conf[m] >= self.tau
            served[m] = s
            e_t = float(err[m][s].mean()) if s.any() else 0.0
            self.pending.append(e_t)
            self.err_sum += e_t
            self.cov_sum += float(s.mean())
            self.t += 1
            if len(self.pending) > self.delay:
                e_old = self.pending.popleft()
                self.tau = float(np.clip(self.tau + self.gamma * (e_old - self.alpha),
                                         self.tau_min, self.tau_max))
        return served

    def long_run(self):
        return {"aci_rounds": self.t,
                "aci_long_run_err": self.err_sum / max(self.t, 1),
                "aci_long_run_cov": self.cov_sum / max(self.t, 1),
                "aci_tau": self.tau}

    def state_dict(self):
        d = dict(self.__dict__)
        d["pending"] = list(self.pending)
        return d

    def load_state_dict(self, d):
        d = dict(d)
        d["pending"] = deque(d["pending"])
        self.__dict__.update(d)
