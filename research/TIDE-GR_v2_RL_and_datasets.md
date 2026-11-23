# TIDE-GR v2 — Adding Reinforcement Learning, Choosing Datasets, Colab Scope

> **Status.** Revision of [TIDE-GR_spec_v1.md](TIDE-GR_spec_v1.md) made on 2026-11-23 after the professor asked for (a) reinforcement learning, (b) a social-media dataset with user past history, (c) a second dataset, (d) running on Google Colab under time pressure.
> Where this file and v1 differ, **this file wins on scope, RL and datasets**; v1 still holds all the equations it does not change.
> Nothing has been run. Every "expected effect" below is a reasoned expectation from the cited literature, not a measured result. The only way to know the metrics is the pilot in §8.

---

## 1. What the evidence says about RL in recommendation (and why it matters for our metrics)

| Finding | Source | Consequence for us |
|---|---|---|
| Adding a Q-learning head next to a supervised head (SQN / SAC) reported HR/NDCG gains on RetailRocket and RC15 | Xin et al., SIGIR 2020 ([arXiv 2006.05779](https://arxiv.org/abs/2006.05779)) | RL *can* help ranking, but only as an auxiliary signal to a supervised model |
| Re-assessments find those gains are fragile; part of the gain may come from better history embeddings, not from learning a policy; simpler non-RL objectives can match it | Rajapakse & Jannach ([project page](https://dilina-r.github.io/rl-rec)); Labarca, Parra & Toro Icarte ("On the Unexpected Effectiveness of RL for Sequential Recommendation") | **Do not** put an RL head into the main ranking loss and expect a reliable NDCG gain. Reviewers know this literature |
| The standard next-item offline protocol hides RL weaknesses and cannot show long-term value | Deffayet et al., SIGIR Forum 2022 ([arXiv 2301.00993](https://arxiv.org/abs/2301.00993)) | Any RL claim must be evaluated with an **unbiased** protocol, not only logged next-item NDCG |
| Exploration (e.g., Thompson sampling) trades immediate accuracy for information; on logged data it looks worse, because logs only reward what the old policy showed | General bandit result; OPE for TS: Jeunen 2025 ([arXiv 2504.08773](https://arxiv.org/abs/2504.08773)) | Exploration will **lower** logged NDCG. It must be a separate serving mode, judged on randomly exposed data |
| Deciding *when / how* to retrain under drift has been formulated as RL | Liebman, Zavesky & Stone, AAMAS 2018 | A clean, defensible place for RL in a continual recommender: the **update controller** |
| Offline deep-RL recommenders (CQL, SQN, DQN) are benchmarked on KuaiRec/KuaiRand | EasyRL4Rec ([arXiv 2402.15164](https://arxiv.org/abs/2402.15164)); DARLR ([arXiv 2505.07257](https://arxiv.org/abs/2505.07257)) | Gives us ready RL baselines on our social-media dataset |

**Conclusion.** RL that sits *inside the ranking objective* is the most likely place to hurt or add noise to NDCG/Recall. RL that sits in the *control loop* (choosing how to adapt) can only change which update the model makes, and it can be made safe by including the v1 behaviour as one of its actions. RL used for *exploration* must be reported separately, because it lowers logged accuracy by design.

---

## 2. Where RL goes in TIDE-GR v2

### R1 — Drift-aware Adaptation Controller (core RL contribution)

Every block, a **contextual Thompson-sampling bandit** chooses *how* the model adapts.

**Context** $x_k\in\mathbb R^6$, all computed before training on the new block (no leakage):

1. mean standardised surprise $\bar S_{k+1}$ (Eq. 27–29 of v1);
2. fraction of active nodes with $D_v>h$ (how many users drifted);
3. popularity divergence $J_{k+1}=\mathrm{JS}(\mathrm{pop}_k\|\mathrm{pop}_{k+1})$;
4. calibration error (ECE) of the last block;
5. share of new users/items in the block;
6. $\log$ block size.

**Actions** (small discrete set, so few rounds suffice):

| Arm | Plasticity $q_1$ | Epochs | Replay : new ratio | Meaning |
|---|---|---|---|---|
| a0 | 0.5 | $E$ | 1 : 1 | **v1 default** (safe arm) |
| a1 | 0.1 | $E/2$ | 2 : 1 | Conservative: protect old knowledge |
| a2 | 1.0 | $E$ | 1 : 2 | Plastic: follow change |
| a3 | 1.0 | $2E$ | 1 : 4 | Very plastic (after strong shift) |
| a4 | — | 0 | — | Skip SGD: only predict step + admit new nodes (cheap) |

**Reward** for the action taken at block $k$, observed after block $k+1$ arrives:

$$
r_k=\mathrm{NDCG@20}(B_{k+1})\;-\;\omega_f\,\big[\mathrm{NDCG@20}_{\text{mem-val}}^{\,k}-\mathrm{NDCG@20}_{\text{mem-val}}^{\,k+1}\big]_+\;-\;\omega_c\,\mathrm{ECE}(B_{k+1})
$$

* term 1 = accuracy on the future block (plasticity);
* term 2 = forgetting, measured on a held-out validation slice of the replay memory (**not** the retention test slices — leakage rule L4 stays intact);
* term 3 = calibration.

**Policy.** Bayesian linear regression per arm, Thompson sampling. The prior mean favours a0, so early blocks behave like v1.

**Why this is valid online learning, not test tuning.** The NDCG for block $k+1$ is recorded *before* the reward is used, and the reward only affects decisions for blocks $k+2$ onward. This is the same prequential logic as ACI in v1.

**Why it should not hurt metrics.** The v1 configuration is one of the arms. The worst case is a few rounds of exploration on a poor arm. The comparison that proves value is **R1 vs. the best single fixed arm chosen in hindsight** (an oracle that a static method cannot actually have).

**Why it fits our story.** In v1, drift only modulated plasticity through a hand-set formula $q_v=q_0+q_1\tanh(D_v/h)$. R1 *learns* the mapping from drift signals to adaptation policy. Uncertainty and drift now drive a learned decision, which strengthens the "uncertainty as an actuator" claim.

**Honest limit.** KuaiRand gives ≈17 daily decision rounds, Amazon ≈20–30 quarterly rounds. That is few. Mitigation: run the bandit per **user drift group** (5 quantile groups of $D_v$, each group gets its own $q_1$ choice), which gives ≈5× more reward observations, with the reward computed on each group's users. If the pilot shows the bandit cannot beat the fixed a0 arm, R1 is reported as a negative result and a0 is used (the paper still stands on v1's contribution).

### R2 — Belief Thompson-sampling exploration mode (uses the Gaussian belief for free)

Exploration mode ranks by a **sample** from the belief instead of its mean:

$$
\tilde\theta_v\sim\mathcal N(\mu_v,\mathrm{diag}(\lambda_v)^{-1}),\qquad \mathrm{rank}^{\text{TS}}_{ui}=\big(P(\pi)\tilde\theta\big)_u^\top\big(P(\pi)\tilde\theta\big)_i+\alpha_u h_\psi(\hat x_i).
$$

* No new parameters. Uncertain nodes (new items, drifted users) get explored automatically, because their λ is low.
* **Accuracy mode (rank by $m$) stays the default** for all headline accuracy numbers.
* Evaluated on KuaiRand's **randomly exposed videos**, which give an unbiased test of preference without the old policy's exposure bias.

### R3 — SQN-style Q-head (optional, ablation only)

An auxiliary Q-learning head trained with KuaiRand's multi-signal reward (long-view, like, follow, share). Because the reproducibility studies in §1 question its gains, it is **excluded from the main model** and run once as an ablation. Keep it only if it improves validation NDCG in the pilot.

---

## 3. Metric impact of each RL piece (the check the professor asked for)

| RL piece | Accuracy (Recall/NDCG@20, logged) | Accuracy on random-exposure (unbiased) | Forgetting / BWT | Calibration (ECE) | Cold-start / coverage | Cost |
|---|---|---|---|---|---|---|
| R1 controller | **Neutral to positive** expected; a0 is always an option, so the floor is ≈v1 minus exploration cost | Neutral to positive | **Expected better**: forgetting is in the reward | Expected better: ECE is in the reward | Neutral | ≈0 (tiny regression model); arm a4 *saves* compute |
| R2 TS exploration | **Expected lower** (by design) | Uncertain; this is what we test | Neutral | Neutral (calibration uses mean mode) | **Expected higher** item coverage and new-item exposure | One extra sample pass |
| R3 Q-head | Contested in the literature: may go up, down, or stay flat | Unknown | Unknown | May worsen (RL head is not calibrated) | Unknown | +10–30% training |

**Rule to protect the paper's headline metrics.** Headline tables use accuracy mode with R1 on. R2 gets its own table on unbiased data. R3 is an ablation row.

---

## 4. Datasets

### Dataset 1 (social media, with past history): **KuaiRand-Pure**

* Source: Kuaishou short-video app; Gao et al., CIKM 2022 ([arXiv 2208.08696](https://arxiv.org/abs/2208.08696), [kuairand.com](https://kuairand.com)).
* Size (per the paper's table; recheck when downloading): **27,285 users, 7,551 videos, ≈1.44 M interactions**. Fits Colab comfortably.
* **Past history:** the standard-feed log for **Apr 8 – Apr 21, 2022** is history *before* the evaluation window; the standard and random logs cover **Apr 22 – May 8**. File names imply ≈31 days in total, but the abstract says "two weeks", so confirm the date ranges on kuairand.com after download.
* **12 feedback signals** (click, like, follow, comment, share, view time…) → a natural RL reward and a social-media setting.
* **Randomly exposed videos** inserted into feeds → unbiased evaluation for R2 and an unbiased check of the main model. Very few datasets have this; it is the strongest reason to pick KuaiRand.
* Item text: a 2026 Zenodo supplement lists video captions and categories ([record 18159199](https://zenodo.org/records/18159199)) → new-item prior (M8). If unusable, fall back to category one-hots.

**Protocol on KuaiRand.**

* Positive = click **and** (long view or like/follow/share). Exact threshold fixed on the history weeks only.
* Warm-up = Apr 8 – Apr 21 (the "past history"); stream = **daily blocks** Apr 22 – May 8 (≈17 blocks).
* ACI rounds = **hourly** (≈400 rounds), so the v1 guarantee stays non-vacuous; feedback delay = same day.
* Environment covariates: season one-hot is meaningless over one month → replace with day-of-week one-hot.

### Dataset 2 — FINAL CHOICE (2026-11-23, 10-hour budget): **Amazon Reviews 2023 Video_Games, official 5-core**

* 94,762 users, 25,612 items, 814,586 ratings (official 5-core page; [amazon-reviews-2023.github.io/data_processing/5core.html](https://amazon-reviews-2023.github.io/data_processing/5core.html)). Toys 5-core is 3.9 M ratings → too slow for a 10-hour budget, so Toys is dropped.
* Use 2015-01 to the end of the data, quarterly blocks; warm-up = 2015–2017, stream = 2018 onward (≈23 blocks; verify the per-quarter counts after download).
* Re-apply the past-only 5-core rule on top of the official file (leakage rule L1); remaining leakage from the official global filter is stated as a limitation.
* The 10-hour run plan is in §5b.

### (Superseded) Dataset 2 (long multi-year history, text for new items): **Amazon Reviews 2023, one category**

* Source: McAuley Lab ([amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io)).
* Why: many years of history → quarterly blocks with real seasonality; rich item text → tests M8; widely used, so reviewers accept it.
* **Colab sizing.** Full Toys_and_Games is several million reviews (a mirror lists ≈8.1 M; check the official card). To stay inside Colab RAM and time: keep a fixed time window (e.g., 2017–2023), apply a **past-only** 5-core filter (leakage rule L1), and cap at roughly 2–3 M interactions. If time is very short, use **Video_Games** (smaller, was already the v1 pilot dataset).
* Blocks = quarters; ACI rounds = weeks (as in v1).

### Datasets considered and rejected

| Dataset | Reason not chosen |
|---|---|
| Yelp | Has a friend graph and long history, but no random exposure and no clear advantage over Amazon; keep as a backup second dataset |
| MicroLens-100K | Official protocol is leave-one-out; timestamps in the public release are unverified |
| Tenrec | ≈5 M users / 140 M interactions: too large for Colab in our time |
| MIND | News; very short item lifetime; impression format needs a different protocol |
| Epinions / Douban | Small and old; weak OOD story |

The pair **KuaiRand (short horizon, fast drift, unbiased exposure, social signals) + Amazon (long horizon, seasonal drift, cold-start text)** covers two very different kinds of shift. That diversity is itself an argument reviewers like.

---

## 5. Colab-feasible scope (we are short of time)

The full v1 has nine modules. To finish in time, v2 splits them into a **core** that is always built and **pilot-gated** parts.

| Module | v2 status | Change from v1 |
|---|---|---|
| M0 stream/environments | Core | Daily blocks on KuaiRand |
| M1 LightGCN encoder | Core | Mean + variance passes; **edge-sample passes dropped** (S=0) unless M2 is kept |
| M1b environment head | Core | Cheap, PDA-style; keep |
| M2 edge reliability | **Simplified** | Fixed age-decay weights $\rho(\Delta)$ as edge weights; the learned edge net only if the pilot shows gain |
| M3 Gaussian belief + Laplace + drift-gated forgetting | Core | Unchanged; this is the main contribution |
| M4 predictive + 3-parameter calibration | Core | $v=v^{\text{node}}$ only when S=0 |
| M5 surprise-CUSUM drift detector | Core | Also feeds the R1 context |
| M6 replay memory | Core | Replay ratio now chosen by R1 |
| M7 selective serving + ACI | Core | Hourly rounds on KuaiRand |
| M8 open-world priors | Core | KuaiRand: captions/categories; Amazon: text |
| V-REx invariance | **Pilot-gated** | Keep only if it beats tuned ERM in the pilot (v1 §16) |
| **R1 RL controller** | **Core (new)** | §2 |
| **R2 TS exploration** | **Core (new), separate table** | §2 |
| R3 Q-head | Ablation only | §2 |

### Updated per-block loop

```text
0  CONTROL : context x_k (drift + calibration + new-node stats) -> Thompson-sample arm a_k (per drift group)
1  TEST    : predict B_{k+1} (accuracy mode; also TS mode for the R2 table); log metrics
2  REWARD  : r_{k-1} for arm a_{k-1} now observable -> update bandit posterior
3  DETECT  : surprise S_u -> CUSUM D_u; global alarm
4  PREDICT : lambda <- (1/lambda + q(D; q1 from a_k))^-1
5  ADMIT   : new users / items with low-precision priors
6  UPDATE  : SGD with epochs and replay ratio from a_k (skipped if a_k = a4)
7  LAPLACE : lambda += curvature of new data
8  FIT     : calibration (kappa, b, nu), text map W, trend eta
9  MEMORY  : select / evict + curvature hand-back
```

### Rough Colab budget (estimate, to be measured)

* KuaiRand-Pure, d=64, L=2: LightGCN fine-tuning on ≈1.4 M edges takes minutes per block on a T4. 17 blocks × (1.3–1.6× LightGCN) → about 1–2 GPU-hours per full run.
* Amazon (≈2–3 M edges, ≈25 blocks): several GPU-hours per run.
* Budget: 5 seeds × (main + ~6 ablations + ~6 baselines) per dataset. That is too many for free Colab. **Use 3 seeds**, Colab Pro if available, and save a checkpoint after every block (v1 §18) so disconnects don't lose work.

---

## 5b. 10-hour-per-dataset run plan (T4, estimates until block 1 is timed)

Requirements on the code: negative sampling on the GPU, sparse propagation, chunked full-ranking evaluation, warm-up checkpoint shared by all ablations of a seed, checkpoint after every block. Epochs are **not** cut (early stopping only), so metrics are not lowered to save time; the time is saved by reusing checkpoints and limiting seeds for secondary rows.

| Item | KuaiRand-Pure | Video_Games 5-core |
|---|---|---|
| Light hyper-parameter check on warm-up (3–4 key params) | 0.5–1 h | 0.5–1 h |
| Warm-up checkpoints, 3 seeds (shared) | 0.3–0.5 h | 0.5–0.8 h |
| TIDE-GR main, 3 seeds (stream only) | 0.8–1.5 h | 1.2–2 h |
| 6 ablations × 1 seed (stream only) | 1.5–2.5 h | 2–3 h |
| Baselines: LightGCN-FT ×3, PDA, GraphSAIL/LWC-KD, SASRec, SQN (KuaiRand only) | 2.5–3.5 h | 2–3 h |
| LightGCN retrain-per-block (most expensive baseline), 1 seed | 1–1.5 h | 1–1.5 h |
| **Total** | **≈7–10 h** | **≈7.5–10 h** |

If the first timed block shows the budget will be exceeded: drop to 2 seeds for the main model before cutting any baseline, then run ablations on the last half of the stream only.

## 5c. Accuracy boosters (pilot-gated, added 2026-11-23)

Plain LightGCN + BPR is not the most accurate static graph CF model. Three training-time additions keep inference linear, so the closed-form variance, Laplace update and drift test are unchanged. Each is kept only if it raises validation NDCG@20 in the pilot.

| Booster | Evidence | Compatibility |
|---|---|---|
| B1 XSimGCL-style noise-contrastive loss on the encoder | SimGCL/XSimGCL report clear Recall@20 gains over LightGCN on sparse benchmarks ([arXiv 2209.02544](https://arxiv.org/abs/2209.02544)) | Training-only term; Laplace curvature still from the ranking loss (approximation A4 extended) |
| B2 Sampled-softmax loss with many negatives instead of BPR | Wu et al., [arXiv 2201.02327](https://arxiv.org/abs/2201.02327): softmax-type losses beat BPR for item recommendation | Diagonal curvature is still closed form (softmax Hessian diagonal) |
| B3 Recent-interest term for KuaiRand: user vector + mean of last-n item vectors | Short-video behaviour is strongly sequential (closes part of the gap to SASRec) | Linear in item embeddings, so variance still propagates |

## 6. Baselines (both datasets, same protocol)

1. LightGCN retrained per block; LightGCN fine-tuned per block.
2. Continual GNN recommenders: GraphSAIL or LWC-KD, and GraphPro (if code runs).
3. OOD: PDA (popularity-drift forecast); InvCF or DR-GNN.
4. Bayesian continual: online EWC on LightGCN (a special case of M3, so a direct ablation).
5. **RL baselines:** SQN on SASRec (Xin et al.); CQL from EasyRL4Rec (KuaiRand only).
6. Uncertainty reference: 5-member ensemble of LightGCN (upper reference, as in v1).

---

## 7. Metrics and the key ablations

**Metrics.**

* Accuracy: Recall@20, NDCG@20 (full ranking).
* Unbiased accuracy: NDCG/AUC on KuaiRand random-exposure items.
* Continual: average accuracy over blocks, BWT (forgetting).
* Calibration: ECE, NLL, risk-coverage curve, realised ACI error.
* Open world: Recall@20 on new users/items vs. number of interactions seen.
* Exploration (R2): item coverage, new-item exposure share.

**Ablations that decide what goes in the paper.**

| Ablation | Question |
|---|---|
| R1 vs. fixed a0 vs. **best fixed arm in hindsight** vs. random arm | Does the RL controller beat the static choice? |
| R1 context without drift features | Is it the drift/uncertainty signal that makes the controller work? |
| −M3 forgetting gate (q fixed) | Value of drift-gated forgetting |
| −M5 (raw-loss CUSUM) | Is surprise better than raw loss (H3)? |
| −M6 memory | Retention, and V-REx needs memory |
| −calibration ν | Fix C1 matters? |
| +R3 Q-head | Does an RL ranking head help or hurt? |

---

## 8. Two-week pilot before the full run (decides the final architecture)

On KuaiRand-Pure only, with M1 + M3 + M5 + R1:

1. Surprise-CUSUM vs. raw-loss CUSUM on synthetic preference swaps (H3).
2. R1 vs. fixed a0 and vs. best fixed arm.
3. V-REx vs. tuned ERM.
4. R3 Q-head vs. no head.

Anything that does not beat its simpler alternative is demoted to an ablation. **This is the only honest way to make sure the final model has the best metrics we can get**: decide from measured numbers, not from expectations.

---

## 9. Claims for the paper (revised)

1. One Gaussian belief per node serves as uncertainty, anti-forgetting weight, plasticity and new-node prior (v1).
2. A surprise-based detector separates lack of evidence from real change (v1, H3).
3. **New:** an RL (contextual bandit) controller learns *how much to adapt* from drift and uncertainty signals, under a reward that balances accuracy, forgetting and calibration, and the v1 rule is a special case of its policy.
4. **New:** the same belief yields Thompson-sampling exploration at no extra cost, evaluated on unbiased random-exposure data.
5. Evaluated on two very different shift regimes: fast social-media drift (KuaiRand) and slow seasonal e-commerce drift (Amazon).
