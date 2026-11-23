# Stage 7b — Experimental Design

Every experiment maps to a hypothesis in `research_gap_and_novelty.md` §8. Data and splits are as in `dataset_analysis.md` §5–6. Baselines are as in `baseline_comparison.md`.

## 0. Global protocol

* **Prequential block evaluation.** Test on $B_{k+1}$ before updating on it, for all $k\ge W_0$. This holds for all methods, including static OOD baselines, which are run in two modes:
  * (a) **retrain-per-block** on all data up to $B_k$;
  * (b) **fine-tune-per-block** from the previous checkpoint.
* **Full ranking** over all items known before $B_{k+1}$, plus new items in the new-item track. Excluded are items the user interacted with in the training history. Full ranking is used rather than sampled metrics (Krichene & Rendle, KDD'20).
* **Global-timeline split.** No random or leave-one-out split for any temporal claim (Ji et al., TOIS'23; Meng et al., RecSys'20).
* **Seeds and tuning.** 5 seeds. Equal tuning budget per method, on the warm-up only.

## A. Recommendation quality (H1)

**Metrics:** Recall@K, NDCG@K, HitRate@K and Precision@K, for K ∈ {10, 20}.

For user $u$ with relevant set $R_u$ in $B_{k+1}$ and ranked list $L_u^K$:

* $\text{Recall@K}=|L_u^K\cap R_u|/|R_u|$
* $\text{Precision@K}=|L_u^K\cap R_u|/K$
* $\text{HR@K}=\mathbb 1[|L_u^K\cap R_u|>0]$
* $\text{NDCG@K}=\text{DCG}/\text{IDCG}$, using binary gains

Metrics are averaged over users in a block, then over blocks. **Per-block curves are also reported.**

## B. Temporal OOD generalisation (H1, H2)

* **Next-block accuracy** $A_{k+1|k}$: the model after $B_k$, evaluated on $B_{k+1}$.
* **Degradation.** $\Delta_{\text{OOD}}=A_{k|k}^{\text{in}}-A_{k+1|k}$, where $A^{\text{in}}_{k|k}$ is accuracy on a held-out in-period slice of $B_k$.
* **Horizon decay.** Evaluate $f_k$ on $B_{k+h}$ for $h=1..4$ *without* updates. This gives the slope of accuracy against horizon.
* **Shift-stratified analysis.** Bin test interactions by:
  * item popularity change $|\log n_i^{k+1}-\log n_i^{k}|$;
  * season (holiday vs. non-holiday quarters).

  Report accuracy per bin.
* **Static-shift comparability.** One run on the official Amazon-2023 absolute-timestamp split, against DR-GNN and DRGO.

## C. Invariant preference learning (H2)

* $\lambda_{\text{inv}}\in\{0,0.1,1,10\}$, with the environment term on and off. This is a 2×4 grid.
* **Cross-environment transfer.** Train with environments $\{1..k\}$ but *remove* all same-season blocks from memory, then test on the next same-season block. This asks whether the model generalises to a recurring environment it has not recently seen.
* **Representation probe.** Linear probe predicting block-level item popularity from the preference factor $\mu_i$ vs. from LightGCN embeddings. Lower probe accuracy means less popularity information leaks into the preference factor.
* **Comparators:** InvCF, CausalDiffRec, DR-GNN, DRGO, PDA, and ERM (λ_inv=0) **tuned with the same budget**.
* **Pre-registered expectation:** gains may be small (Gulrajani & Lopez-Paz'21; Wild-Time'22). A null result is reported.

## D. Structural uncertainty (H7)

Perturbations are injected into the training graph of a chosen block:

1. **Random noise edges** (5%, 10%, 20% of edges), uniform over (u, i).
2. **Popularity-biased fake edges** (to trending items): realistic exposure noise.
3. **Stale edges**: duplicate old interactions with a fresh timestamp. These are old preferences made to look recent.
4. **Edge removal** (10–30%): sparsification.

**Metrics:**

* AUROC and AUPRC of edge entropy / $1-\pi_{ui}$ for detecting injected edges.
* Accuracy degradation vs. perturbation rate, with and without M2.
* Correlation between node $v^{\text{edge}}$ and the fraction of perturbed edges.

**Baselines:** RGCF (denoising), T-CE/R-CE, DR-GNN reweighting, AdvDrop masks, and random scores.

## E. Predictive uncertainty (H5, H6)

The calibration target is $y_{ui}=1$ if $u$ interacts with $i$ in $B_{k+1}$. Evaluated pairs are positives plus $n_{\text{neg}}=99$ uniformly sampled negatives drawn from items active in $B_{k+1}$. **This defines the base rate (1%), and it is stated in every table.**

* **ECE (expected calibration error).** Bin $p$ into 15 equal-mass bins: $\text{ECE}=\sum_b \frac{|b|}{N}|\text{acc}(b)-\text{conf}(b)|$. Lower is better. Adaptive binning is used because of the low base rate.
* **Brier score** $=\frac1N\sum(p-y)^2$, reported with its reliability/resolution decomposition.
* **NLL** $=-\frac1N\sum[y\log p+(1-y)\log(1-p)]$. Appropriate because $p$ is a probability over this defined candidate distribution.
* **AUROC for error detection.**
  * Item level: does $v_{ui}$ (or $1-p$ for positives) separate false positives in the top-K from true positives?
  * User level: does $c_u$ separate users with HR@K=0 from those with HR@K=1?
* **Risk–coverage curves.** Sort users by confidence. For coverage $c$, report the selective risk = 1 − HR@K among the top-$c$ fraction. Summarise with AURC; lower is better.
* **Online coverage.** The realised selective error per block vs. the target α, with and without ACI.
* **Track breakdown:** warm, drifted (CUSUM-flagged), new-user and new-item.

**Uncertainty baselines on the same backbone:**

* score-only confidence;
* MC dropout;
* deep ensemble (M=5; Video_Games only, as upper reference);
* temperature scaling and Kweon'22 Gaussian calibration (static);
* evidential head (GPN-style; optional).

## F. Continual learning (H4)

Let $a_{t,j}$ be the accuracy (NDCG@20) on held-out test data of block $j$ after training through block $t$.

* **Average performance (prequential):** $\bar A=\frac1{T}\sum_t a_{t,t+1}$.
* **Average accuracy (retention view):** $\text{ACC}_T=\frac1T\sum_{j\le T}a_{T,j}$.
* **Forgetting:** $F_j=\max_{t<T}a_{t,j}-a_{T,j}$, averaged over $j$.
* **Backward transfer (Lopez-Paz & Ranzato, NeurIPS'17):** $\text{BWT}=\frac1{T-1}\sum_{j<T}(a_{T,j}-a_{j,j})$.
* **Adaptation (plasticity):** $a_{t,t}$ (in-block, after update) and steps-to-recover after a CUSUM alarm.
* **Seasonal recall:** accuracy on block $j+4$ (same season) relative to $a_{j,j}$.
* **Memory:** bytes for the precision vectors (≈ the size of the embeddings) plus the replay buffer, and the comparison to each baseline.
* **Stability–plasticity frontier:** plot $\bar F$ against $\bar A$ across $q_1$ and $\lambda_\phi$ values for TIDE-GR, uniform online-Laplace (q constant), EWC, replay-only, fine-tune and full retrain.
* **Synthetic drift (Video_Games):** detection delay, false-alarm rate, and AUROC of $D_v$ for injected users. Compared detectors: raw loss + CUSUM, raw $v$, embedding displacement $\|\mu^{k+1}-\mu^k\|$, and MMD of user-neighbourhood embeddings.

## G. Open-world adaptation (H8)

* **New users:** reveal $n\in\{0,1,3,5,10\}$ interactions, then rank the rest of $B_{k+1}$. Report Recall/NDCG@20 and ECE by $n$.
* **New items:** NDCG@20 restricted to new items, and the share of new items in top-K (coverage).
* **Evidence curve:** accuracy and mean $v$ vs. the number of interactions. Expected: $v$ decreases monotonically and accuracy increases.
* **Insertion cost:** wall-clock per new node, and per 1k new nodes, vs. retraining.

**Baselines:** INMO, IDCF (if it scales), DropoutNet, ALDI, CLCRec, TDRO, and the GraphPro / GPT4Rec handling of new nodes.

## H. Ablation studies

| Variant | Tests |
|---|---|
| − M1b environment term | Value of the observable-environment factorisation |
| λ_inv = 0 | Value of invariance (H2) |
| Time-block envs → EIIL-inferred envs | Environment definition |
| − M2 (all π=1) | Structural uncertainty (H7) |
| M2 prior constant (no time decay) | Temporal staleness prior |
| q_v ≡ q_0 (no drift gating) | Coupling detector → plasticity (H4) |
| q ≡ 0 and fixed λ (≈ online EWC) | Bayesian forgetting at all |
| Replace M3 + M5 with "EWC + separate MC dropout + raw-loss ADWIN" at equal compute | **Coupling ablation: the central novelty test (R1)** |
| Uniform reservoir instead of reliability/drift priority | M6 policy |
| No memory | Interaction of invariance and memory (penalty degenerates) |
| − ACI (static τ) | Online calibration (H5) |
| β_r = 0 vs. > 0 | Risk-averse ranking |
| New-node prior: zero mean / high precision | M8 components |
| Surprise without the epistemic term (probit shrinkage off) | Mechanism of evidence vs. change (H3) |

## I. Efficiency

Report per block:

* training time;
* adaptation time;
* PREDICT_ALL time;
* inference latency per user (p50 / p95, CPU and GPU);
* peak GPU memory;
* state size;
* scaling with $|E|$, using 25 / 50 / 100% subsamples of Toys.

All measurements on fixed hardware, stated.

## J. Statistical reliability

* **Seeds:** 5 per configuration, covering initialisation and negative sampling. Splits are fixed and released.
* **Tests.**
  * Per-block metrics across methods: **paired** comparison. Use a Wilcoxon signed-rank test over (block × seed) units, or a paired t-test on per-user metrics within blocks with Holm–Bonferroni correction across baselines.
  * Calibration metrics: 1,000-resample user-level bootstrap.
* **Confidence intervals:** 95% bootstrap CIs on all headline numbers. Effect sizes are reported, not only p-values.
* **Fair tuning:** the same trial budget and search procedure for all methods, on the warm-up only. Official code with authors' defaults as the starting point (Shehzad & Jannach, RecSys'23; Dacrema et al., RecSys'19).

## 10. Leakage and contamination controls

1. **No global k-core and no global ID remapping that uses future data.** Filtering uses past data only.
2. **Item statistics** (popularity, age) used in $x^e$ are computed from data strictly before the prediction time. The forecast $\hat x^{k+1}$ uses only $B_{\le k}$.
3. **Normalisation constants** (degrees, CUSUM null calibration) come from the past only.
4. **Validation slices** are the time-last 10% of the training block, never the test block.
5. **Retention evaluation sets** are carved out once and never trained on, including by replay.
6. **Text encoder contamination.** A frozen pretrained text encoder may have seen product text from after a block's date. Mitigations:
   * use an encoder released before the start of the stream window, or
   * run an ablation without text (ID-only new-item prior via the population mean) and report the gap.
7. **Benchmark contamination of baselines.** Do not reuse baseline papers' reported numbers on different splits. Rerun everything under our protocol.
8. **Duplicate reviews and re-listings** (variants of the same product): use parent_asin.

## 11. Mapping hypotheses → experiments

| Hypothesis | Experiments |
|---|---|
| H1 | A, B |
| H2 | B (stratified), C, ablations |
| H3 | F (synthetic drift), ablations |
| H4 | F (frontier), ablations |
| H5 | E (calibration, online coverage) |
| H6 | E (risk–coverage) |
| H7 | D |
| H8 | G |

## 12. Compute budget estimate (to be validated)

* **Video_Games prototype:** a single consumer GPU. Order-of-magnitude estimate: tens of minutes per block, about 30 blocks, 5 seeds.
* **Toys_and_Games:** about 4–10× more.
* **Deep-ensemble reference:** Video_Games only.

These are planning estimates, not measurements.
