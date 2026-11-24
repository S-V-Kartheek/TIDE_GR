"""R1: drift-aware adaptation controller = contextual linear Thompson-sampling bandit.

One decision per drift group (quintiles of the drift score D among the users
of the block being learned). Context x (7 features + bias). Reward observed one
block later: NDCG_g(next block) - w_f * forgetting - w_c * ECE_g, centred across
groups of the same block (removes the block-level difficulty).
Per arm a: Bayesian linear regression  w_a ~ N(A^-1 b, sigma^2 A^-1).
Arm 0 is the v1 default; its prior mean gets a +delta0 bonus so early blocks
behave like v1 (safe start).
"""
from __future__ import annotations

import numpy as np

ARM_DEFAULTS = {"q1": 0.5, "epoch_mult": 1.0, "replay_ratio": 1.0, "anchor_mult": 1.0, "lr_mult": 1.0}

DEFAULT_ARMS = [
    {"name": "a0_default", "q1": 0.5, "epoch_mult": 1.0, "replay_ratio": 1.0},
    {"name": "a1_conservative", "q1": 0.1, "epoch_mult": 0.5, "replay_ratio": 2.0},
    {"name": "a2_plastic", "q1": 1.0, "epoch_mult": 1.0, "replay_ratio": 0.5},
    {"name": "a3_very_plastic", "q1": 1.0, "epoch_mult": 2.0, "replay_ratio": 0.25},
    {"name": "a4_skip", "q1": 0.5, "epoch_mult": 0.0, "replay_ratio": 1.0},
]


class LinTSBandit:
    def __init__(self, arms=None, dim: int = 8, prior_std: float = 0.05, noise_std: float = 0.05,
                 delta0: float = 0.01, seed: int = 0):
        self.arms = list(arms or DEFAULT_ARMS)
        self.dim = dim
        self.sigma2 = noise_std ** 2
        self.A = [np.eye(dim) * (self.sigma2 / prior_std ** 2) for _ in self.arms]
        self.b = [np.zeros(dim) for _ in self.arms]
        self.delta0 = float(delta0)
        self.rng = np.random.default_rng(seed)
        self.pending: list = []  # decisions waiting for reward
        self.history: list = []

    def choose(self, x: np.ndarray) -> int:
        vals = []
        for a in range(len(self.arms)):
            Ainv = np.linalg.inv(self.A[a])
            mu = Ainv @ self.b[a]
            w = self.rng.multivariate_normal(mu, self.sigma2 * Ainv)
            vals.append(float(x @ w) + (self.delta0 if a == 0 else 0.0))
        return int(np.argmax(vals))

    def update(self, arm: int, x: np.ndarray, r: float):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += x * r

    def record(self, block: int, group_users: list, contexts: list, arms: list, forget: float = 0.0):
        self.pending.append({"block": block, "users": group_users, "x": contexts, "arms": arms,
                             "forget": forget})

    def set_forgetting(self, value: float):
        if self.pending:
            self.pending[-1]["forget"] = float(value)

    def reward_from(self, per_user_ndcg: dict, per_user_ece_fn, w_f: float, w_c: float,
                    min_users: int = 20):
        """Resolve all pending decisions with the metrics of the newly tested block."""
        done = []
        for dec in self.pending:
            rs, idx = [], []
            for g, users in enumerate(dec["users"]):
                vals = [per_user_ndcg[u] for u in users if u in per_user_ndcg]
                if len(vals) < min_users:
                    continue
                ece_g = per_user_ece_fn([u for u in users if u in per_user_ndcg])
                r = float(np.mean(vals)) - w_f * max(0.0, dec["forget"]) - w_c * ece_g
                rs.append(r)
                idx.append(g)
            if rs:
                rs = np.asarray(rs)
                rc = rs - rs.mean() if len(rs) > 1 else rs * 0.0
                for g, r in zip(idx, rc):
                    self.update(dec["arms"][g], np.asarray(dec["x"][g]), float(r))
                self.history.append({"block": dec["block"], "arms": dec["arms"],
                                     "rewards": rs.tolist()})
            done.append(dec)
        self.pending = [d for d in self.pending if d not in done]

    def state_dict(self):
        return {"A": self.A, "b": self.b, "rng": self.rng.bit_generator.state,
                "pending": self.pending, "history": self.history, "arms": self.arms}

    def load_state_dict(self, d):
        self.A, self.b = d["A"], d["b"]
        self.rng.bit_generator.state = d["rng"]
        self.pending, self.history, self.arms = d["pending"], d["history"], d["arms"]


class PairedTSBandit:
    """v2 controller: contextual linear Thompson sampling on *paired* gains.

    Reward of arm a at block k = (NDCG@20 of the model updated with arm a) - (NDCG@20 of the model updated
    with the reference arm 0), both measured on the SAME users of block k+1 (the updates start from the same
    state, "shadow" models). The pairing removes the block difficulty and the user-group difficulty that made
    the v1 reward unlearnable, so the reference arm has gain 0 by definition and only arms 1..K-1 are
    modelled:  gain_a(x) = x . w_a,  w_a ~ N(A_a^-1 b_a, sigma^2 A_a^-1).
    The bandit plays the arm with the largest sampled gain if it is > 0, otherwise the reference arm
    (a safe fallback: an arm is only used when the evidence says it beats the default).
    feedback = "all": every arm has a shadow model, so every arm is observed every block (no exploration cost);
    feedback = "chosen": only the played arm and the reference are observed (classic bandit feedback).
    """

    def __init__(self, arms, dim: int = 8, prior_std: float = 0.3, noise_std: float = 0.3,
                 seed: int = 0, warm_blocks: int = 2):
        self.arms = [{**ARM_DEFAULTS, **a} for a in arms]
        self.dim = dim
        self.sigma2 = noise_std ** 2
        self.A = [np.eye(dim) * (self.sigma2 / prior_std ** 2) for _ in self.arms]
        self.b = [np.zeros(dim) for _ in self.arms]
        self.rng = np.random.default_rng(seed)
        self.warm_blocks = int(warm_blocks)
        self.n_decisions = 0
        self.pending: list = []
        self.history: list = []

    def gains(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        g = np.zeros(len(self.arms))
        for a in range(1, len(self.arms)):
            Ainv = np.linalg.inv(self.A[a])
            mu = Ainv @ self.b[a]
            w = self.rng.multivariate_normal(mu, self.sigma2 * Ainv) if sample else mu
            g[a] = float(x @ w)
        return g

    def choose(self, x: np.ndarray) -> int:
        self.n_decisions += 1
        if self.n_decisions <= self.warm_blocks:
            return 0
        return int(np.argmax(self.gains(x)))

    def update(self, arm: int, x: np.ndarray, gain: float):
        if arm == 0:
            return
        self.A[arm] += np.outer(x, x)
        self.b[arm] += x * gain

    def state_dict(self):
        return {"A": self.A, "b": self.b, "rng": self.rng.bit_generator.state, "pending": self.pending,
                "history": self.history, "arms": self.arms, "n_decisions": self.n_decisions}

    def load_state_dict(self, d):
        if "n_decisions" not in d or [a["name"] for a in d["arms"]] != [a["name"] for a in self.arms]:
            return  # state of a different controller (e.g. a v1 warm-up checkpoint): start fresh
        self.A, self.b = d["A"], d["b"]
        self.rng.bit_generator.state = d["rng"]
        self.pending, self.history = d["pending"], d["history"]
        self.arms = [{**ARM_DEFAULTS, **a} for a in d["arms"]]
        self.n_decisions = d["n_decisions"]
