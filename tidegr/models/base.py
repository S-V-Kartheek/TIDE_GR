"""Common method interface used by the prequential runner.

Protocol per stream block k (the runner calls, in order):
  prepare_eval(k)            cache representations for predicting block k
  score / score_rel / score_ts / pair_mv   (test on block k, before training on it)
  observe_test(k, info)      drift detection, RL reward (TIDE-GR only)
  update(k, ctx)             learn from block k (train split)
  prepare_eval(k)            refresh after the update (for calibration on block k val)
  finalize_block(k)          make block k's validation edges part of the past
"""
from __future__ import annotations

import numpy as np
import torch


class Method:
    supports_var = False
    supports_ts = False
    supports_rel = False

    def __init__(self, cfg, data, device, seed):
        self.cfg, self.data, self.device, self.seed = cfg, data, device, int(seed)

    # --- learning
    def warmup(self, ctx):
        raise NotImplementedError

    def update(self, block: int, ctx):
        raise NotImplementedError

    def finalize_block(self, block: int):
        pass

    def observe_test(self, block: int, info: dict):
        pass

    # --- prediction
    def prepare_eval(self, block: int, light: bool = False):
        raise NotImplementedError

    def score(self, users: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score_rel(self, users: torch.Tensor) -> torch.Tensor:
        return self.score(users)

    def score_ts(self, users: torch.Tensor) -> torch.Tensor:
        return self.score(users)

    def pair_mv(self, u: np.ndarray, i: np.ndarray):
        """Mean score (and variance or None) for explicit (user, item) pairs."""
        raise NotImplementedError

    # --- state
    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, st: dict):
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}
