# TIDE-GR

**T**emporally-**I**nvariant, **D**rift-gated, **E**vidence-calibrated **G**raph **R**ecommendation with an RL adaptation controller.

One shared codebase runs TIDE-GR and every baseline on both datasets under the same prequential (test-then-train) protocol. It is built for Google Colab T4, and every run can resume on **any** Colab account from the last finished block.

* Architecture specification: `research/TIDE-GR_FINAL.md` (design docs are in the original research folder).
* Datasets: **KuaiRand-Pure** (short-video social media; Apr 8–21 history, Apr 22–May 8 daily stream, random-exposure log for unbiased evaluation) and **Amazon Reviews 2023 Video_Games 5-core** (history before 2018, quarterly stream from 2018).

---

## 1. Run on Colab (recommended)

1. In Google Drive, create one folder (e.g. `TIDE_GR_runs`) and share it as **Editor** with your other Google accounts. On every other account: right-click the shared folder → *Organize → Add shortcut to My Drive*. The path is then `/content/drive/MyDrive/TIDE_GR_runs` on every account.
2. Open `notebooks/colab_kuairand.ipynb` (or `colab_amazon_vg.ipynb`) in Colab → *Runtime → Change runtime type → T4 GPU*.
3. Set `RUNS_ROOT` in the settings cell and run all cells:
   * **Quick check** (~15–30 min): builds the dataset, runs TIDE-GR / LightGCN-FT / SASRec on a short stream, checks metrics and resume, and prints the estimated GPU-hours of the full plan.
   * **Full plan**: tuning on history only, then TIDE-GR + baselines + seeds + ablations.
4. When a session ends (time limit or GPU quota), open the notebook on another account and run all cells again. Finished runs are skipped and the interrupted run continues from its last checkpoint.
5. The last cell writes the paper tables: `<RUNS_ROOT>/<dataset>/results/{summary.csv, table.md, significance.csv}`.

If the GitHub repo is private, add a Colab secret `GITHUB_TOKEN` (a token with read access).

## 2. Command line

```bash
pip install -r requirements.txt
# quick check (timing + sanity + resume test)
python -m tidegr.quick_check --data data/kuairand.yaml --runs_root <RUNS_ROOT>
# full plan (resumable)
python -m tidegr.run_all --plan plans/kuairand.yaml --runs_root <RUNS_ROOT> --max_hours 11
python -m tidegr.run_all --plan plans/amazon_vg.yaml --runs_root <RUNS_ROOT> --max_hours 11
# only some runs
python -m tidegr.run_all --plan plans/kuairand.yaml --runs_root <RUNS_ROOT> --only tidegr,lightgcn_ft
# H3 pilot: surprise-CUSUM vs raw-loss CUSUM on 5% synthetic preference swaps
python -m tidegr.run_all --plan plans/kuairand_h3.yaml --runs_root <RUNS_ROOT>
# tables + Wilcoxon/Holm significance vs TIDE-GR
python -m tidegr.report --runs_root <RUNS_ROOT> --dataset kuairand
# local smoke test of everything (CPU, a few minutes)
python -m tidegr.run_all --plan plans/synthetic.yaml --runs_root ./runs --device cpu
```

## 3. Checkpointing and resume

* `<RUNS_ROOT>/<dataset>/<run>/seed<k>/state.pt` holds the full state: model, belief (μ, λ), graph, drift CUSUMs, RL bandit posteriors, replay memory, calibration, ACI threshold, metrics so far and all RNG states.
* It is written atomically (tmp file → rename, previous copy kept as `state.pt.prev`) after every stream block and every 5 warm-up epochs. A crash during a save cannot corrupt it.
* A time guard (`max_hours`, default 11 h) stops cleanly before the Colab limit.
* `<RUNS_ROOT>/<dataset>/status.json` shows done / running / failed for every run.
* Tested: a run interrupted 33 times and resumed in new processes produced the same NDCG, Recall, BWT and unbiased AUC as an uninterrupted run.

## 4. What is in the code

| Paper component | Code |
|---|---|
| Common data format, splits (train / time-last 10% val / 5% retention), leakage rules | `tidegr/data/stream.py`, loaders `kuairand.py`, `amazon.py` |
| Cumulative graph, age-decayed edge weights, LightGCN mean / variance passes | `tidegr/graph.py` |
| TIDE-GR and LightGCN-family baselines (one configurable class) | `tidegr/models/graphrec.py` |
| Gaussian belief, Laplace precision (BPR / sampled-softmax GGN), drift-gated forgetting | `graphrec.py` (`_curvature`, `_predict_step`) |
| Environment (trend) head + forecast, V-REx on common users, XSimGCL contrast, recent-interest term | `graphrec.py`, `tidegr/features.py` |
| Surprise-CUSUM drift detector + global alarm | `tidegr/drift.py` |
| RL controller: contextual Thompson-sampling bandit (R1) | `tidegr/rl_bandit.py` |
| Thompson-sampling exploration mode (R2) | `graphrec.py` (`score_ts`) |
| Replay memory with seasonal slots and curvature hand-back | `tidegr/memory.py` |
| Open-world priors (neighbour mean / precision-weighted text ridge) | `tidegr/openworld.py`, `graphrec.py` (`_admit`) |
| 3-parameter calibration, delayed-feedback ACI | `tidegr/calibration.py` |
| Prequential runner, all metrics, checkpoint / resume | `tidegr/runner.py`, `tidegr/eval/metrics.py` |
| History-only tuning (coordinate search on a mini-stream) | `tidegr/tune.py` |
| SASRec, SQN baselines | `tidegr/models/sasrec.py` |
| Deep ensemble reference | `tidegr/models/ensemble.py` |

**Configs.** `configs/base.yaml` ← `configs/data/*.yaml` ← `configs/methods/*.yaml` ← plan overrides ← tuned parameters. Plans (`configs/plans/*.yaml`) list every run in priority order.

**Metrics per stream block:**
* Recall / NDCG @10/@20 with full ranking, in three modes: accuracy, reliability (m − β√(νv)) and Thompson.
* ECE, NLL, Brier.
* AURC, selective error, coverage and long-run ACI error.
* Retention recall and BWT.
* New-user and new-item recall.
* Unbiased AUC / NDCG@10 on the random-exposure log.
* Drift-detection AUC (H3 pilot).

## 5. Honest notes for the paper

* `kd_lite` is a uniform L2-anchor stand-in for distillation baselines (GraphSAIL / LWC-KD). Name it as such, or replace it with the official code if time allows.
* CQL (EasyRL4Rec) is not included. The RL baselines here are SQN (Q-head on SASRec) and our controller's own ablations (random arm, fixed arms).
* The learned edge-reliability network (ablation A9) is not implemented. Edge reliability uses the age prior ρ(Δ) plus test-time Bernoulli edge sampling.
* XSimGCL contrast and V-REx are switched on only if history-only tuning finds they help. On the synthetic test the contrast term hurt, so its default is off.
* The belief prior is expressed in data units (λ₀ = n₀ × curvature of one interaction). Without this, the prior dominated the evidence; see `TIDE-GR_FINAL.md` §4.4 and the code comments.
