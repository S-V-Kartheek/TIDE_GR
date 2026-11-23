"""Execute an experiment plan (tuning + all runs x seeds) with resume.

  python -m tidegr.run_all --plan configs/plans/kuairand.yaml \
      --runs_root /content/drive/MyDrive/TIDE_GR_runs [--only tidegr,lightgcn_ft] [--max_hours 11]

Re-running the same command (on any Colab account that mounts the same shared
folder) skips finished runs, resumes the unfinished one from its last
checkpoint and continues with the rest. Progress: <runs_root>/<dataset>/status.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

from .compose import _load_file, _p, compose, tuned_params
from .data.registry import load_data
from .runner import Runner
from .tune import tune_method
from .utils import (TimeBudget, TimeLimitReached, atomic_json_save, get_device, load_json, log,
                    set_log_file)


def run_plan(plan_path: str, runs_root: str | None = None, only: list | None = None,
             max_hours: float | None = None, skip_tune: bool = False, device_pref: str = "auto"):
    plan = _load_file(_p(plan_path))
    data_yaml = plan["data"]
    data_ov = plan.get("data_overrides") or {}
    probe = compose(data_yaml, plan["runs"][0]["method"], extra=data_ov)
    runs_root = runs_root or probe.get("runs_root", "./runs")
    max_hours = float(max_hours if max_hours is not None else probe.get("max_hours", 11.0))
    budget = TimeBudget(max_hours)
    device = get_device(device_pref)
    data = load_data(probe, runs_root)
    ds_name = plan.get("results_name", data.name)
    ds_dir = os.path.join(runs_root, ds_name)
    set_log_file(os.path.join(ds_dir, "run_all_log.txt"))
    log(f"[plan] {plan_path} dataset={data.name} device={device} budget={max_hours}h data={data.summary()}")
    status_path = os.path.join(ds_dir, "status.json")
    status = load_json(status_path, {})
    for run in plan["runs"]:
        name = run["name"]
        if only and name not in only:
            continue
        seeds = run.get("seeds", [0])
        try:
            tuned = None
            src = run.get("tuned_from", name if run.get("tune") else None)
            if src:
                tuned = tuned_params(runs_root, plan.get("tuning_from_dataset", ds_name), src)
                if tuned is None and run.get("tune") and not skip_tune:
                    status[name] = {"state": "tuning", "updated": time.ctime()}
                    atomic_json_save(status, status_path)
                    tuned = tune_method(data, data_yaml, run["method"], name, runs_root, device, budget,
                                        overrides=run.get("overrides"), extra=data_ov)
                elif tuned is None and src != name:
                    log(f"[plan] {name}: tuned params of '{src}' not available yet; using defaults")
            for s in seeds:
                key = f"{name}/seed{s}"
                if status.get(key, {}).get("state") == "done":
                    continue
                cfg = compose(data_yaml, run["method"], run.get("overrides"), tuned=tuned, extra=data_ov)
                run_dir = os.path.join(ds_dir, name, f"seed{s}")
                status[key] = {"state": "running", "updated": time.ctime(), "dir": run_dir}
                atomic_json_save(status, status_path)
                log(f"[plan] >>> {key}")
                r = Runner(cfg, data, run_dir, seed=s, device=device, max_hours=budget.remaining() / 3600)
                summ = r.run()
                status[key] = {"state": "done", "updated": time.ctime(), "dir": run_dir,
                               "ndcg@20": summ.get("acc_ndcg@20"), "hours": summ.get("wall_hours")}
                atomic_json_save(status, status_path)
        except TimeLimitReached:
            log("[plan] session time budget reached. Everything is saved. "
                "Re-run the same command (any account, same shared folder) to continue.")
            atomic_json_save(status, status_path)
            return status
        except Exception:
            log(f"[plan] run {name} FAILED:\n{traceback.format_exc()}")
            status[name] = {"state": "failed", "updated": time.ctime(), "error": traceback.format_exc()[-2000:]}
            atomic_json_save(status, status_path)
            if os.environ.get("TIDEGR_STRICT"):
                raise
    log("[plan] all requested runs processed")
    return status


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--runs_root", default=None)
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--max_hours", type=float, default=None)
    ap.add_argument("--skip_tune", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    only = [x.strip() for x in a.only.split(",")] if a.only else None
    run_plan(a.plan, a.runs_root, only, a.max_hours, a.skip_tune, a.device)


if __name__ == "__main__":
    sys.exit(main())
