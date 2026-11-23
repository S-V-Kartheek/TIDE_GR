"""Smoke tests on the synthetic stream (CPU, ~2 min):  python -m pytest -q tests"""
import os
import shutil
import tempfile

import numpy as np
import torch

from tidegr.compose import compose
from tidegr.data.registry import load_data
from tidegr.runner import Runner
from tidegr.utils import TimeLimitReached

FAST = {"budget_margin_s": 0, "method.epochs.warmup_max": 12, "method.epochs.warmup_eval_every": 1,
        "method.epochs.warmup_patience": 100, "method.epochs.stream": 2, "method.ckpt_every_epochs": 3}


def _run(method, root, overrides=None, max_hours=None, name="run"):
    cfg = compose("data/synthetic.yaml", method, overrides={**FAST, **(overrides or {})})
    data = load_data(cfg, root)
    return Runner(cfg, data, os.path.join(root, name), seed=0, device=torch.device("cpu"),
                  max_hours=max_hours).run()


def test_all_methods_run():
    root = tempfile.mkdtemp()
    try:
        for m in ["tidegr", "lightgcn_ft", "xsimgcl_ft", "pda", "ewc", "kd_lite"]:
            s = _run(f"methods/{m}.yaml", root, name=m)
            assert np.isfinite(s["acc_ndcg@20"]) and s["acc_ndcg@20"] > 0, m
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_resume_is_exact():
    root = tempfile.mkdtemp()
    try:
        a = _run("methods/tidegr.yaml", root, name="A")
        b = None
        for _ in range(300):
            try:
                b = _run("methods/tidegr.yaml", root, max_hours=15 / 3600, name="B")
                break
            except TimeLimitReached:
                continue
        assert b is not None
        for k in ["acc_ndcg@20", "acc_recall@20", "bwt_recall@20"]:
            assert abs(a[k] - b[k]) < 1e-6, (k, a[k], b[k])
    finally:
        shutil.rmtree(root, ignore_errors=True)
