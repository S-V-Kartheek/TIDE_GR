# TIDE-GR — FINAL Architecture (locked for coding)

**T**emporally-**I**nvariant, **D**rift-gated, **E**vidence-calibrated **G**raph **R**ecommendation with an RL adaptation controller

> **Status (2026-11-23).** This file is the **single source of truth for coding, running and the paper**. It supersedes `TIDE-GR_spec_v1.md`, `TIDE-GR_v2_RL_and_datasets.md`, `final_architecture.md` and `algorithm_pseudocode.md` wherever they differ. Equation numbers "(v1 Eq. n)" point to `TIDE-GR_spec_v1.md`, which still holds the derivations.
> Nothing has been run yet. Every module below is **on by default**. Each has a config switch, so the pilot and the ablations can turn it off. The pilot (§9) only removes a module if it measurably lowers validation NDCG@20 or ECE.

---

## 1. One-paragraph summary

A LightGCN recommender whose user and item embeddings are **Gaussian beliefs** (mean μ, precision λ). The belief is trained with a strong ranking loss (sampled softmax + light graph-contrastive noise), updated block by block on a time-ordered stream, and kept from forgetting by a **precision-weighted anchor** to its previous value. A **surprise-based drift detector** finds users whose behaviour really changed and **loosens only their beliefs**. A **contextual-bandit RL controller** learns, from drift and calibration signals, how strongly to adapt each block. The same belief gives **calibrated probabilities**, **selective (abstaining) recommendations**, **low-confidence priors for new users and items**, and **Thompson-sampling exploration** for free.

## 2. Contributions claimed in the paper

1. **One belief, four jobs**: a per-node Gaussian belief is simultaneously the uncertainty estimate, the anti-forgetting weight, the plasticity control and the new-node prior.
2. **Evidence-vs-change detection**: a standardised-surprise CUSUM that separates "little data" from "taste changed", and gates forgetting per node.
3. **RL adaptation controller**: a contextual Thompson-sampling bandit that learns how much to adapt from drift/uncertainty signals, rewarded on next-block accuracy, forgetting and calibration. The hand-set v1 rule is one of its arms.
4. **Reliable serving under drift**: calibrated probabilities + selective recommendation with an online (ACI) threshold, and belief-Thompson exploration evaluated on unbiased random-exposure data.
5. **Prequential evaluation on two shift regimes**: fast social-media drift (KuaiRand) and slow seasonal e-commerce drift (Amazon Video_Games).

**Not claimed:** causal identification of preferences; a new GNN encoder; per-user guarantees.

---

## 3. Final module list (all locked)

| # | Phase | Final choice | Default | Alternatives rejected (reason) |
|---|---|---|---|---|
| P0 | Data | Time blocks; past-only filtering; positives defined per dataset (§7) | — | Random split (leaks future); global k-core (leakage) |
| P1 | Graph | Cumulative bipartite graph; edge weight = age prior ρ(Δ) (v1 Eq. 3) | on | Learned edge-reliability net (cost, instability; kept as ablation A9) |
| P2 | Encoder | LightGCN, L=2 (tune {2,3}), d=64, mean pass + variance pass (v1 Eq. 6–7) | on | NGCF (non-linear, breaks closed form); TGN/DyGFormer (5× cost, no gain shown for top-K) |
| P2b | Accuracy booster B1 | XSimGCL-style noise contrastive loss (§4.3) | on | SGL (graph augmentation, slower) |
| P2c | Accuracy booster B3 | Recent-interest term (§4.2) | on for KuaiRand, n=20; on for Amazon, n=5 | SASRec encoder (no closed-form uncertainty; kept as baseline) |
| P3 | Environment head | α_u·h_ψ(x_i^e) with trend forecast (v1 Eq. 8–9) | on | DRO adversary (overlaps; extra optimisation) |
| P4 | Ranking loss (B2) | Sampled softmax, 64 negatives, temperature τ_s (§4.3) | on | BPR (weaker ranking loss; kept as ablation A8) |
| P5 | Invariance | V-REx on common users (v1 Eq. 12–13), weight tuned in {0, 0.1, 1} on warm-up | on | IRM (unstable), GroupDRO (pessimistic) |
| P6 | Belief + anti-forgetting | Diagonal Gaussian; precision-weighted anchor; Laplace precision update (v1 Eq. 11, 16–17; curvature for softmax §4.4) | on | EWC (special case = ablation), distillation (uniform, not evidence-weighted) |
| P7 | Drift detector | Standardised surprise + per-node CUSUM + global alarm (v1 Eq. 25–29) | on | Raw-loss CUSUM (ablation A3), ADWIN |
| P8 | Drift-gated forgetting | λ ← (1/λ + q(D))⁻¹, q₁ chosen by the RL controller (v1 Eq. 14–15) | on | Fixed q (ablation A2) |
| P9 | **RL controller (R1)** | Contextual Thompson-sampling bandit, 5 arms, per drift group (§5) | on | SQN/SAC RL head (fragile gains; ablation A10), DQN/PPO (too few rounds) |
| P10 | Prediction | Closed-form score variance (node) + 4 Bernoulli edge-sample passes at test time only; 3-parameter probit calibration (v1 Eq. 18–24) | on | Deep ensemble (5× cost; upper reference only), MC dropout |
| P11 | Serving | Accuracy mode (headline); reliability mode m−β√v; selective serving with ACI threshold (v1 Eq. 33–35) | on | Fixed threshold; split conformal (needs exchangeability) |
| P12 | **Exploration (R2)** | Belief Thompson sampling (§5.3), separate table | separate mode | ε-greedy (ignores uncertainty) |
| P13 | Memory | Environment-stratified A-Res reservoir + seasonal slots + curvature hand-back (v1 Eq. 36) | on | Uniform replay |
| P14 | Open world | New user: ζ·mean of first items; new item: W·text (ridge, v1 Eq. 30–32); low precision + local MAP | on | DropoutNet/CLCRec (separate training, no uncertainty) |

---

## 4. Model details that are new or changed since v1

### 4.1 Representations

Propagated mean and variance, with edge weights $r_{ui}=\rho(\Delta_{ui})$:

$$
m=P(r)\,\mu,\qquad s=P^{\circ 2}(r)\,\lambda^{-1}.
$$

### 4.2 User representation with recent interest (B3)

Let $R_u$ be the last $n$ items of user $u$ before the prediction time:

$$
\tilde m_u=m_u+\gamma\,\frac1{|R_u|}\sum_{i\in R_u}m_i,\qquad
\tilde s_u=s_u+\frac{\gamma^2}{|R_u|^2}\sum_{i\in R_u}s_i,\qquad \gamma\in\{0,0.5,1\}\ (\text{tuned on warm-up}).
$$

Score: $s(u,i,e)=\tilde m_u^\top m_i+\alpha_u h_\psi(x_i^e)$.

### 4.3 Training loss (B2 + B1)

Sampled softmax over the positive and $n_s=64$ negatives $j\sim$ uniform over items active in the block:

$$
\ell^{\text{ssm}}(u,i)=-\log\frac{\exp(s(u,i)/\tau_s)}{\exp(s(u,i)/\tau_s)+\sum_{j}\exp(s(u,j)/\tau_s)},\qquad \tau_s\in\{0.1,0.2\}.
$$

XSimGCL noise contrast (training only): add $\epsilon=0.2$ uniform noise of fixed norm to each layer's embeddings, contrast layer-1 view vs. final view with InfoNCE (temperature 0.2) over the batch; weight $\lambda_{cl}\in\{0.05,0.1,0.2\}$.

Full per-block objective (sum scale, then divided by $N_k$):

$$
\mathcal L_k=\sum_{e}\sum_{(u,i)\in D_e}\ell^{\text{ssm}}_e(u,i)
+\lambda_{\text{inv}}N_k\,\mathrm{Var}_e[R^c_e]
+\lambda_{cl}N_k\,\mathcal L^{\text{cl}}
+\tfrac12\sum_v\|\theta_v-\mu_v^{\text{pred}}\|^2_{\lambda_v^{\text{pred}}}
+\tfrac12\lambda_\phi\|\phi-\phi^{(k-1)}\|^2 .
$$

### 4.4 Laplace curvature for the softmax loss

For one training pair with candidate logits $x_c=\hat z_u^\top\hat z_c/\tau_s$ and softmax probabilities $p_c$, keep the diagonal of the GGN:

$$
G_u\mathrel{+}=\frac{1}{\tau_s^2}\sum_c p_c(1-p_c)\,\hat z_c^{\circ2},\qquad
G_c\mathrel{+}=\frac{1}{\tau_s^2}\,p_c(1-p_c)\,\hat z_u^{\circ2}.
$$

Then $\lambda_v\leftarrow\lambda_v^{\text{pred}}+\gamma_F\big(P^{\circ2\top}G\big)_v$ as in v1 Eq. 17. The contrastive and V-REx terms do not contribute curvature (stated approximation A4).

### 4.5 Calibration target and drift test

Unchanged from v1: binary target "interacted in the next block" over positives + 99 sampled negatives; 3-parameter probit calibration (κ, b, ν); standardised surprise uses those calibrated binary probabilities, so the ranking-loss change does not affect the detector.

---

## 5. Reinforcement learning

### 5.1 R1 — adaptation controller

* **Decision time:** start of every block, before training; one decision per drift group $g\in\{1..5\}$ (quintiles of $D_u$ among active users).
* **Context** $x_{k,g}\in\mathbb R^7$: group mean surprise, group share with $D_u>h$, global JS popularity divergence, last block ECE, share of new users, share of new items, log block size. Standardised with warm-up statistics.
* **Arms:**

| Arm | q₁ (plasticity) for the group | Epochs | Replay : new |
|---|---|---|---|
| a0 (v1 default) | 0.5 | E | 1 : 1 |
| a1 conservative | 0.1 | E/2 | 2 : 1 |
| a2 plastic | 1.0 | E | 1 : 2 |
| a3 very plastic | 1.0 | 2E | 1 : 4 |
| a4 skip | — | 0 (predict step + admission only) | — |

Epochs and replay ratio are block-level: they are taken from the arm chosen for the largest group. q₁ is per group.

* **Reward** (observed after block $k+1$):

$$
r_{k,g}=\mathrm{NDCG@20}_g(B_{k+1})-\omega_f\big[\mathrm{NDCG@20}^{k}_{\text{mem-val}}-\mathrm{NDCG@20}^{k+1}_{\text{mem-val}}\big]_+-\omega_c\,\mathrm{ECE}_g(B_{k+1}),\quad \omega_f=1,\ \omega_c=0.5 .
$$

* **Policy:** Bayesian linear regression per arm (prior $\mathcal N(0,I)$, noise variance from warm-up), Thompson sampling; prior mean of a0 raised by $\delta_0=0.01$ so early blocks behave like v1.
* **Warm-start:** during warm-up, run each arm once on the last 4 warm-up blocks (counterfactual replays from the same checkpoint) to seed the posteriors. This uses only warm-up data.
* **Leakage rule:** the metric for block $k+1$ is logged before $r_k$ is used; $r_k$ only affects decisions from block $k+2$.

### 5.2 Why this cannot lower the headline metrics much

a0 is the v1 behaviour; the prior favours it; exploration is limited to a few rounds. The paper compares R1 against a0, a random arm, and the **best fixed arm in hindsight**.

### 5.3 R2 — belief Thompson-sampling exploration

$\tilde\theta\sim\mathcal N(\mu,\mathrm{diag}(\lambda)^{-1})$, rank by $\big(P\tilde\theta\big)_u^\top\big(P\tilde\theta\big)_i+$ environment term. Reported only in the exploration table, evaluated on KuaiRand random-exposure logs (unbiased NDCG/AUC), item coverage and new-item exposure.

### 5.4 R3 — SQN-style Q-head

Ablation A10 only: a Q-head on the user representation with reward = weighted KuaiRand feedback (long view 1, like 2, follow 3, share 3). Not part of the final model.

---

## 6. Per-block algorithm (final)

```text
WARM-UP (history blocks; each block = one environment):
  init mu ~ N(0, 0.01), lambda = lambda0, phi
  tune {L, lambda_inv, lambda_cl, tau_s, gamma, lambda0, q0} on warm-up validation (small grid, §8)
  train L_k on all warm-up blocks; Laplace precision; fit W, (kappa,b,nu), eta; fill memory; C = 0; tau = tau0
  seed bandit posteriors with counterfactual arm replays on the last 4 warm-up blocks

FOR each stream block k -> k+1:
  0 CONTROL : context per drift group -> Thompson-sample arm a_{k,g}
  1 TEST    : cached passes -> accuracy-mode lists, reliability-mode lists, TS-mode lists, p, v; log all metrics;
              hourly/weekly ACI updates of tau
  2 REWARD  : rewards r_{k-1,g} are now observable -> update bandit posteriors
  3 DETECT  : surprise S_u -> CUSUM C_u, D_u; global alarm
  4 PREDICT : lambda_v <- (1/lambda_v + q_v)^-1 with q1 from the group's arm
  5 ADMIT   : new users (neighbour mean), new items (W * text); low precision; local MAP
  6 UPDATE  : minimise L_{k+1} on new block + memory (epochs, replay ratio from the arm; skip if a4)
  7 LAPLACE : select memory edges; lambda += gamma_F * curvature of the rest
  8 FIT     : W, (kappa,b,nu) on validation slice (before the next DETECT); if alarm: refit psi, eta
  9 MEMORY  : insert with quotas; evicted edges hand curvature back to lambda
 10 RETAIN  : evaluate on held-out slices of past blocks (metrics only)
  checkpoint the full state
```

---

## 7. Datasets and protocol

| | KuaiRand-Pure (social media) | Amazon Reviews 2023 Video_Games, 5-core |
|---|---|---|
| Size | 27,285 users · 7,551 videos · 1.44 M logged impressions | 94,762 users · 25,612 items · 814,586 ratings |
| Positive | `is_click`=1 **and** (`long_view`=1 or like/follow/share/comment); thresholds fixed on history weeks | every review |
| History (warm-up) | standard log Apr 8 – Apr 21 | 2015-01 – 2017-12 |
| Stream | daily blocks Apr 22 – May 8 (≈17) | quarterly blocks 2018-Q1 onward (≈23) |
| Unbiased test | random-exposure log (never trained on) | — |
| ACI rounds | hourly | weekly |
| Item text (new-item prior) | captions/categories (Zenodo supplement); fallback: category one-hot | title + description |
| Season covariate | day-of-week one-hot | quarter one-hot |

Protocol rules (all from v1 §14): past-only filtering; covariates from past data; validation = last 10% (by time) of each training block; retention slices never trained on; full ranking over all items active so far; frozen text encoder released before the stream (e.g., all-MiniLM-L6-v2).

---

## 8. Hyper-parameters (final defaults and the only tuned ones)

Tuned on warm-up only, same budget for every method (≈1 h per dataset):

| Param | Grid |
|---|---|
| L | {2, 3} |
| λ_inv (V-REx) | {0, 0.1, 1} |
| λ_cl (contrast) | {0.05, 0.1, 0.2} |
| τ_s (softmax temperature) | {0.1, 0.2} |
| γ (recent interest) | {0, 0.5, 1} |
| λ₀ (prior precision) | {0.1, 1, 10} |

Fixed: d=64; lr 1e-3 (Adam); batch 4096; n_s=64 negatives; early stop on validation NDCG@20 (patience 5); E=20 stream epochs; ρ₀=0.95, ρ∞=0.6, τ_ρ=2 blocks; q₀=1e-3; δ=0.5, h=4, ς=0.9; γ_F=1; memory 2% per environment, W_M=8; ζ=1, T_loc=20; λ_W=1; β_r=0.5 (reliability mode); n_neg=99 for calibration; S=4 edge samples (test only); α (ACI target) = 0.10, γ_aci=0.05.

---

## 9. Pilot (first ≈2 days of compute, KuaiRand)

Run the warm-up + first 5 stream days. For each switch, keep it if it does not lower validation NDCG@20 (and for P7, if it beats raw loss on the synthetic swap test):

1. B1 contrast, B2 softmax, B3 recent interest (accuracy boosters).
2. Surprise-CUSUM vs. raw-loss CUSUM on 5% synthetic preference swaps (H3).
3. R1 vs. a0 (directional check only; the full comparison is in the main run).
4. Time one block → confirm the 10-hour budget.

The final config is frozen after the pilot and used unchanged on both datasets.

---

## 10. Baselines

| Group | Methods |
|---|---|
| Static per block | LightGCN fine-tuned; LightGCN retrained; XSimGCL fine-tuned |
| Continual graph rec | GraphSAIL or LWC-KD; online EWC on LightGCN |
| OOD / popularity drift | PDA |
| Sequential | SASRec (fine-tuned per block) |
| RL | SQN on SASRec; CQL (EasyRL4Rec, KuaiRand only) |
| Uncertainty reference | 5-member LightGCN ensemble (upper reference, 1 seed) |

---

## 11. Metrics

| Family | Metrics |
|---|---|
| Accuracy (headline) | Recall@20, NDCG@20, full ranking, per block and averaged over the stream |
| Unbiased (KuaiRand) | NDCG@10 / AUC on random-exposure items |
| Continual | Average accuracy, BWT (forgetting) |
| Calibration and reliability | ECE (15 bins), NLL, Brier; risk–coverage AURC; realised long-run ACI error |
| Open world | Recall@20 for new users/items vs. #interactions seen |
| Exploration (R2) | Item coverage, new-item exposure share, unbiased NDCG |
| Cost | Wall-clock per block, peak GPU memory |

Statistics: 3 seeds for main model and key baselines; paired Wilcoxon over users with Holm correction.

---

## 12. Ablations

| ID | Change | Tests |
|---|---|---|
| A1 | Remove belief (point embeddings, no anchor) | Value of the belief overall |
| A2 | Fixed q (no drift gating) | Drift-gated forgetting |
| A3 | Raw-loss CUSUM | Surprise statistic (H3) |
| A4 | No memory | Replay + invariance |
| A5 | λ_inv = 0 | Invariance |
| A6 | ν = 0 (Platt) | Variance-aware calibration |
| A7 | R1 → fixed a0 / random arm / best fixed arm in hindsight | RL controller |
| A8 | BPR instead of softmax; no contrast; γ = 0 | Accuracy boosters |
| A9 | Learned edge-reliability net | Structural uncertainty |
| A10 | + SQN Q-head | RL inside the ranking loss |
| A11 | Context without drift features | Whether uncertainty/drift signals drive the RL gains |

---

## 13. Compute plan (T4, per dataset ≤10 h; estimates until block 1 is timed)

| Item | KuaiRand | Video_Games |
|---|---|---|
| Warm-up tuning | 0.5–1 h | 0.5–1 h |
| Warm-up checkpoints × 3 seeds | 0.3–0.5 h | 0.5–0.8 h |
| Main × 3 seeds | 1–1.8 h | 1.4–2.4 h |
| Ablations A1–A8, A11 × 1 seed (A9, A10 KuaiRand only) | 2–3 h | 2–3 h |
| Baselines | 3–4 h | 3–4 h |
| **Total** | **≈7–10 h** | **≈7.5–10 h** |

Overrun rule: main model 3 → 2 seeds first; then ablations on the second half of the stream. Never cut epochs.

---

## 14. Code layout (to implement next)

```text
tidegr/
  configs/            base.yaml, kuairand.yaml, amazon_vg.yaml, ablations/*.yaml
  data/
    kuairand.py       load logs, positive rule, daily blocks, random-exposure test set
    amazon.py         load 5-core, quarterly blocks, item text
    stream.py         Block objects, past-only filtering, covariates, splits, new-node lists
  graph.py            cumulative sparse graph, edge ages, propagate(X, weights, squared, transpose)
  model/
    belief.py         mu, lambda, predict step, admission, checkpoint
    encoder.py        LightGCN mean / variance / sampled passes, recent-interest term
    env_head.py       h_psi, a_u, trend forecast
    losses.py         sampled softmax, XSimGCL contrast, V-REx (common users), Gaussian anchor
    laplace.py        GGN diagonal for softmax, curvature hand-back
    predictive.py     m, v, calibrated p, uncertainty split, ranking modes, Thompson mode
  drift.py            surprise, CUSUM, global alarm, synthetic swap injector
  rl/
    bandit.py         linear Thompson-sampling bandit (R1)
    sqn_head.py       Q-head (A10 only)
  calibration.py      (kappa, b, nu) fit, ACI controller
  memory.py           stratified A-Res + seasonal slots + eviction log
  openworld.py        neighbour prior, weighted ridge W, local MAP
  train.py            warm-up and stream loop (§6), checkpoint per block
  eval/
    metrics.py        Recall, NDCG, ECE, NLL, AURC, BWT, coverage
    unbiased.py       random-exposure evaluation (KuaiRand)
    stats.py          Wilcoxon + Holm
  baselines/          lightgcn_ft.py, lightgcn_retrain.py, xsimgcl.py, ewc.py, graphsail.py, pda.py, sasrec.py, sqn.py, ensemble.py
notebooks/
  colab_kuairand.ipynb, colab_amazon_vg.ipynb   (mount Drive, install, run config, resume from checkpoint)
```

**Build order:** data + graph + metrics + LightGCN-FT baseline → belief + Laplace + predictive + calibration → drift → memory + losses (softmax, contrast, V-REx) → RL bandit → open world → remaining baselines → Colab notebooks → pilot.

---

## 15. Expected outcome (honest)

| Metric | Expectation vs. the strongest baselines | Confidence |
|---|---|---|
| ECE / NLL, risk–coverage | Better | High |
| Forgetting (BWT) | Better | High |
| New-user/item recall | Better or equal | Medium |
| Next-block Recall/NDCG | Competitive to better, with B1–B3 on | Medium |
| R1 vs. best fixed arm | Equal to slightly better | Medium–low |

Target venues: CIKM, RecSys, ECIR, PAKDD (full or short); workshop as a fallback. Approximations A1–A6 and proof obligation PO1 (v1 §15) are stated in the paper's limitations.

---

## 16. Implementation notes (v1.1, code in github.com/S-V-Kartheek/TIDE_GR, 2026-11-23)

These decisions were made while implementing and testing the code. They take precedence over the sections above.

| # | Change | Reason |
|---|---|---|
| I1 | **Prior precision in data units.** λ₀ = n₀ × median curvature of one interaction, set after warm-up; n₀ is tuned in {0.1, 1}. Warm-up training uses standard LightGCN weight decay; the precision-weighted anchor starts in the stream. | With an absolute λ₀ (=10), the curvature of real evidence was ~1% of λ₀: the prior dominated, λ did not separate sparse from active nodes, and plasticity vanished. After the fix, real KuaiRand gives median λ ≈ 0.06 (low-degree users) vs 0.44 (high-degree), with λ₀ = 0.018. |
| I2 | **Reliability ranking and Thompson sampling in calibrated units.** Ranking uses m − β√(ν·v), with ν the fitted calibration scale. TS draws use θ̃ = μ + s·ε/√λ, with s set so that the posterior std of well-evidenced nodes ≈ ½ the typical \|μ\|. | Raw Laplace variances are on a different scale from scores; ν = 0 correctly disables the variance penalty. |
| I3 | **RL decision point.** The arm is chosen after DETECT on the block just tested (the context needs that block's surprise), before the predict step. The reward comes from the next test. Rewards are centred across drift groups of the same block. | Leak-free, and removes block-level difficulty from the reward. |
| I4 | **RL prior.** Bandit prior std 0.01 and a0 bonus 0.02, so the controller starts at the v1 rule and leaves it only on evidence. | Safe start. |
| I5 | **XSimGCL contrast is off by default.** λ_cl ∈ {0, 0.1} is tuned. | On the synthetic test the contrast term (even without noise) reduced Recall@20 from 0.29 to 0.10. It is kept only if history-only tuning shows a gain on the real data. |
| I6 | **Tuning.** Coordinate search on a history-only mini-stream (last 3 history blocks, warm-up capped at 40 epochs). The TIDE-GR grid has 9 trials: loss, τ_s, λ_cl, γ_recent, λ_inv, n₀, L. Each baseline has 1–3 trials. | Fits the 10 h / dataset budget. |
| I7 | **Not implemented.** The learned edge-reliability net (A9) and the SQN head on TIDE-GR (A10). The age prior ρ(Δ) and test-time Bernoulli edge sampling are kept. CQL is not included. `kd_lite` (uniform L2 anchor) stands in for GraphSAIL / LWC-KD. | Time budget; stated as limitations. |
| I8 | **Negative sampling.** Negatives are drawn uniformly over items known so far, not only items active in the environment. | Simpler and standard; it affects all methods equally. |
| I9 | **Amazon partial quarters.** Trailing quarters with < 25% of the median stream-block size are dropped (2023Q2 and Q3 are partial in the crawl). | Avoids near-empty test blocks. |
| I10 | **Retention slices.** 5% of each block is held out and never trained on. BWT is computed on stream blocks. | As specified (L4). |
