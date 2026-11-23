"""Deep ensemble of K GraphRec members (uncertainty upper reference).
Score = member mean; variance = across-member variance (+ calibrated like TIDE-GR)."""
from __future__ import annotations

import numpy as np
import torch

from ..config import merge_cfg
from .base import Method
from .graphrec import GraphRec


class Ensemble(Method):
    supports_var = True
    supports_rel = True

    def __init__(self, cfg, data, device, seed):
        super().__init__(cfg, data, device, seed)
        K = int(cfg.method.get("members", 5))
        mcfg = merge_cfg(cfg, {"method": {"kind": "graphrec"}})
        self.members = [GraphRec(mcfg, data, device, seed * 100 + k) for k in range(K)]
        self.beta_r = float(cfg.method.get("beta_r", 0.5))
        self.done_warm = [False] * K

    def warmup(self, ctx):
        for k, m in enumerate(self.members):
            if self.done_warm[k]:
                continue
            m.warmup(ctx)
            self.done_warm[k] = True
            ctx.save()
        return {}

    def update(self, block, ctx):
        for m in self.members:
            m.update(block, ctx)
        return {}

    def finalize_block(self, block):
        for m in self.members:
            m.finalize_block(block)

    def prepare_eval(self, block, light=False):
        for m in self.members:
            m.prepare_eval(block, light=True)

    @torch.no_grad()
    def _stack(self, users):
        return torch.stack([m.score(users) for m in self.members])

    def score(self, users):
        return self._stack(users).mean(0)

    def score_rel(self, users):
        s = self._stack(users)
        return s.mean(0) - self.beta_r * s.std(0)

    def pair_mv(self, u, i):
        ms = np.stack([m.pair_mv(u, i)[0] for m in self.members])
        return ms.mean(0), ms.var(0)

    def state_dict(self):
        return {"members": [m.state_dict() for m in self.members], "done_warm": self.done_warm}

    def load_state_dict(self, st):
        for m, s in zip(self.members, st["members"]):
            m.load_state_dict(s)
        self.done_warm = st["done_warm"]
