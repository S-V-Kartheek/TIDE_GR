# Stage 2b — Closest Competing Methods

Verification legend for "Read":

* **FT** = methodology and experiment sections read (arXiv HTML or proceedings).
* **Abs** = abstract or metadata only.
* **NV** = not verified.

All numbers quoted are as reported by the authors. They were **not reproduced**. Where a reported number looked internally inconsistent, this is flagged.

Search date: 2026-11-21. Sources: arXiv, ACM DL / DBLP metadata, NeurIPS / PMLR / AAAI / IJCAI proceedings pages, GitHub.

---

## 1. Coverage matrix

Columns are the challenges in our research question. "Temporal Shift" = YES only if the method *models* time (not merely evaluated on a time split).

| # | Paper | Year | OOD | Invariant Preference | Temporal Shift | Structural Uncertainty | Predictive Uncertainty | Continual Learning | Open-World |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DR-GNN (Wang et al., WWW) | 2024 | YES | NO | PARTIAL | PARTIAL | NO | NO | NO |
| 2 | DRGO (Zhao et al., WWW) | 2025 | YES | NO | PARTIAL | PARTIAL | NO | NO | NO |
| 3 | CausalDiffRec (Zhao et al., WWW) | 2025 | YES | YES | PARTIAL | PARTIAL | NO | NO | NO |
| 4 | InvCF (Zhang et al., WWW) | 2023 | YES | YES | PARTIAL | NO | NO | NO | NO |
| 5 | AdvDrop (Zhang et al., WWW) | 2024 | YES | YES | PARTIAL | PARTIAL | NO | NO | NO |
| 6 | PDA (Zhang et al., SIGIR) | 2021 | PARTIAL | PARTIAL | YES | NO | NO | NO | NO |
| 7 | TDRO (Lin et al., AAAI) | 2024 | YES | NO | YES | NO | NO | NO | YES |
| 8 | GraphPro (Yang et al., WWW) | 2024 | NO | NO | YES | NO | NO | PARTIAL | PARTIAL |
| 9 | GPT4Rec (Zhang et al., SIGIR) | 2024 | NO | PARTIAL | PARTIAL | NO | NO | YES | PARTIAL |
| 10 | DEGC (He et al., WWW) | 2023 | NO | NO | PARTIAL | NO | NO | YES | NO |
| 11 | PIW (Wang et al., AAAI) | 2023 | NO | NO | PARTIAL | NO | NO | YES | NO |
| 12 | EvoOOD (Sun et al., TPAMI, accepted) / EAGLE (NeurIPS 2023) | 2025/2023 | YES | YES | YES | NO | NO | NO | NO |
| 13 | ELBO_TDS (Zhu et al., preprint) | 2025 | YES | PARTIAL | YES | NO | PARTIAL | PARTIAL | NO |
| 14 | BGCN-Rec (Sun et al., KDD) | 2020 | NO | NO | NO | YES | PARTIAL | NO | NO |
| 15 | CILER (Wang et al., preprint) | 2026 | YES | PARTIAL | PARTIAL | NO | PARTIAL | NO | NO |
| 16 | Kurle et al., Bayesian CL for non-stationary data (ICLR) — *not recsys* | 2020 | NO | NO | YES | NO | YES | YES | NO |
| 17 | Confidence Gate Theorem (Doku, preprint) — *not GNN* | 2026 | PARTIAL | NO | PARTIAL | PARTIAL | YES | NO | PARTIAL |

### Evidence for every YES / PARTIAL

**1. DR-GNN.** Verified from the full text.

* OOD YES: KL-ball DRO over each user's neighbour distribution. It is evaluated on popularity, temporal and exposure shift.
* Temporal PARTIAL: ML-1M and Food per-user timestamp splits are used for evaluation, but time is not modelled. On ML-1M temporal it is beaten by APDA (NDCG@20 0.2330 vs 0.2362).
* Structural PARTIAL: the closed-form worst case P*(v) ∝ P_u(v)·exp(g/α) acts as adversarial edge reweighting, and the method adds edges. This is worst-case, not a probabilistic uncertainty.

**2. DRGO.** Verified from the full text.

* OOD YES: Sinkhorn DRO with entropy regularisation on group weights.
* Temporal PARTIAL: tested on Food only.
* Structural PARTIAL: a VGAE plus latent diffusion denoises embeddings.
* Some reported improvement percentages are inconsistent between the text and the tables.

**3. CausalDiffRec.** Verified from the full text.

* Invariant YES: variational environment inference, an RL environment generator producing K perturbed graphs, and a variance-based invariance loss, backed by a backdoor-adjustment argument.
* Temporal PARTIAL: Food only.
* Structural PARTIAL: generated perturbed graphs.
* Ablation shows it drops below LightGCN without its generator or inference modules.

**4. InvCF.** Verified from the full text.

* Separate preference and popularity encoders, a popularity-swap consistency loss, and distance-correlation disentanglement.
* Temporal PARTIAL: a Douban temporal split is used for evaluation only.

**5. AdvDrop.** Verified from the full text.

* Invariance: InfoNCE between bias-aware and bias-mitigated graph views.
* Structural PARTIAL: a learned per-edge bias score drives edge dropout. It is trained adversarially and is not a calibrated probability.

**6. PDA.** Verified from the full text.

* Temporal YES: popularity is forecast linearly from the last two stages, m̃ = m^T + α(m^T − m^{T−1}), and used at inference.
* Invariant PARTIAL: the deconfounded matching term is trained with popularity factored out.
* MF only; new users and items are excluded.

**7. TDRO.** Verified from the full text.

* Temporal YES: a "shifting factor" extrapolates the trend of per-period gradients (β_e = exp(p·e)).
* Open-world YES: cold-start items represented through features.
* Not graph-based in its core; backbones are CLCRec and GAR.

**8. GraphPro.** Read the abstract and experiment section.

* Temporal YES: a temporal prompt plus a snapshot protocol.
* Continual PARTIAL: adapts per snapshot but reports no forgetting metrics.
* Open-world PARTIAL: handles new nodes via prompt edges.

**9. GPT4Rec.** Read the abstract and experiment section.

* Continual YES: node-, structure- and view-level prompts on a frozen backbone, compared against GraphSAIL, SGCT, LWC-KD and DEGC.
* Invariant PARTIAL: disentangled views, but no invariance objective.

**10. DEGC.** Read the abstract and experiment section.

* Continual YES: prunes, refines and expands graph convolutions.
* Temporal PARTIAL: separates stale short-term preferences from long-term ones.

**11. PIW.** Abstract only.

* Per-user imitation weight: users whose interests shift imitate the old model less.
* Temporal PARTIAL: models per-user drift only through this weight.

**12. EAGLE / EvoOOD.** Abstract only.

* Environment inference on dynamic graphs, plus invariant patterns via node-wise interventions.
* Evaluated on dynamic link prediction, not top-K recommendation. Not continual.

**13. ELBO_TDS.** Abstract only.

* Temporal generalisation via resampling of time-varying factors and a variational objective.
* Continual PARTIAL: designed for a periodic incremental-training pipeline.
* Predictive PARTIAL: probabilistic model, but no calibration evaluation found.
* Not a graph model.

**14. BGCN-Rec.** Abstract and metadata.

* Structural YES: treats the observed graph as one sample from a random-graph posterior, with node-copying.
* Predictive PARTIAL: Bayesian marginalisation exists, but it is used for accuracy and diversity, not calibration.

**15. CILER.** Abstract only.

* User-conditioned exponential-family latent environments with identifiability claims.
* Predictive PARTIAL: risk-aware marginalisation over environments.

**16. Kurle et al.** Abstract and proceedings.

* Diagonal-Gaussian online variational Bayes plus a raw-data memory.
* "Bayesian forgetting" via a Gaussian diffusion (process-noise) operator for non-stationary streams.
* Not recommendation.

**17. Doku.** Abstract only.

* Shows confidence-based abstention helps when uncertainty is *structural* (cold-start) but can fail under *contextual* temporal drift.
* MovieLens with three shifts. Not a GNN.

---

## 2. Per-paper analysis (the eight that matter most)

### 2.1 DR-GNN — *Distributionally Robust Graph-based Recommendation System* (WWW 2024)

| | |
|---|---|
| **Problem** | GNN recommenders degrade when the train/test distributions differ. |
| **Challenges addressed** | A (statically), partly C-structural. |
| **Challenges remaining** | B (no invariance), C-predictive, D, E. |
| **Architecture** | LightGCN viewed as graph-smoothing regularisation, with per-user KL-DRO over neighbours; Graph Edge-Addition widens support. |
| **Experimental limitations** | Temporal shift is a single timestamp split; temporal gains are weak (Food +3%). LightGCN backbone only. |
| **Adaptable?** | Yes. The worst-case reweighting could serve as an alternative to our edge-reliability module, but it gives no posterior. |
| **Novelty threat** | Low for our combination; **mandatory baseline**. |

### 2.2 DRGO and CausalDiffRec (WWW 2025, same group)

| | |
|---|---|
| **Problem** | DRGO: KL-DRO over-weights noisy samples. CausalDiffRec: environment confounders cause unstable correlations. |
| **Challenges addressed** | A, B (CausalDiffRec), C-structural (partially, via denoising and generated environments). |
| **Challenges remaining** | C-predictive, D, E. The temporal setting is static. |
| **Architecture** | VGAE latent, conditional diffusion, Sinkhorn DRO (DRGO) or environment generator plus invariance loss (CausalDiffRec). |
| **Experimental limitations** | One temporal dataset (Food). Heavy (diffusion steps, RL generator). CausalDiffRec ablations fall below LightGCN when its modules are removed. Reported averages are not reconcilable with the tables. |
| **Adaptable?** | Environment inference is relevant. Diffusion denoising is too costly for repeated continual updates. |
| **Novelty threat** | **Medium** on "invariant OOD graph rec"; we must *not* claim invariant graph rec itself as our contribution. Both are mandatory baselines. |

### 2.3 InvCF (WWW 2023) and PDA (SIGIR 2021)

| | |
|---|---|
| **Problem** | Popularity shift (InvCF) and popularity drift over time (PDA). |
| **Key idea we inherit** | Factorise the score into a *preference* term and a *popularity/exposure* term. PDA additionally shows that **forecasting** the environment term at inference helps when drift is smooth. |
| **Remaining** | No uncertainty, no continual updates, no cold-start. PDA is MF-only and has a linear forecaster. |
| **Novelty threat** | Low; we build on the decomposition. Both are baselines (PDA as the temporal-popularity baseline). |

### 2.4 TDRO (AAAI 2024)

| | |
|---|---|
| **Problem** | Temporal feature shift for **cold-start items**. |
| **Addresses** | A, E, and temporal trend. |
| **Remaining** | Not a graph model; no invariance across environments, no uncertainty, no continual retention. |
| **Novelty threat** | **Medium** for the "temporal + cold-start" sub-claim. We must show what the graph and uncertainty add. Baseline for the cold-item track. |

### 2.5 GraphPro (WWW 2024), GPT4Rec (SIGIR 2024), DEGC (WWW 2023), PIW (AAAI 2023)

| | |
|---|---|
| **Problem** | Streaming / continual graph recommendation. |
| **Addresses** | D (and the A-adaptation part). PIW and DEGC distinguish shifting users from stable ones heuristically. |
| **Remaining** | No OOD generalisation to the *next* period before adaptation, no invariance objective, no uncertainty, no calibration. |
| **Experimental protocol** | Recall/NDCG@20 averaged over future segments (Taobao2014/2015, Netflix, Foursquare; GraphPro: Taobao/Koubei/Amazon). **We should adopt this protocol for comparability.** |
| **Novelty threat** | **Medium** for the continual component. The novelty must not be "a continual graph recommender"; it must be *how uncertainty and invariance govern the update*. GPT4Rec, DEGC and GraphPro are mandatory baselines. |

### 2.6 EAGLE / EvoOOD / DIDA (dynamic-graph OOD)

| | |
|---|---|
| **Problem** | OOD generalisation on dynamic graphs with evolving environments. |
| **Addresses** | A, B, temporal. |
| **Remaining** | Not recommendation (link prediction on generic dynamic graphs); not continual; no calibrated uncertainty. |
| **Novelty threat** | Low–medium. We cite them as the graph-learning counterpart and could use EAGLE's environment-inference idea. |

### 2.7 ELBO_TDS (preprint, 2025)

| | |
|---|---|
| **Problem** | Temporal distribution generalisation in an industrial incremental-training recommender. |
| **Novelty threat** | **The closest framing of "temporal OOD + incremental training"** for recommendation. However, it is not graph-based, has no structural uncertainty or calibration, and has no open-world module. It is unreviewed and its data are proprietary. |

### 2.8 BGCN-Rec (KDD 2020) and Kurle et al. (ICLR 2020)

| | |
|---|---|
| **Relevance** | These two supply the *technical substrate* of our proposal: random-graph (edge) posteriors for GNN recommendation, and Bayesian continual learning with explicit adaptive forgetting. |
| **Novelty threat** | Low. Neither combines the two, neither is temporal-OOD recommendation, and neither evaluates calibration under drift. |

---

## 3. Combinations

We checked whether a **combination** of existing pieces already gives our system:

| Combination | Status |
|---|---|
| DRO/invariant graph rec + continual streaming graph rec | **Not found.** No paper evaluates DR-GNN/InvCF-style objectives inside a GraphSAIL/DEGC/GPT4Rec stream. |
| Continual graph rec + calibrated uncertainty | **Not found** (three independent search passes). |
| Bayesian continual learning (VCL / online Laplace / Kurle) + recommendation GNN | **Not found.** The closest are the Collaborative Kalman Filter (Gultekin & Paisley, arXiv 1501.05624; MF-based, Brownian-motion embeddings, mean-field VI) and BAKE (WWW 2026, continual *knowledge-graph* embedding as sequential Bayesian inference; ACM DOI 10.1145/3774904.3792692; **abstract only**). |
| Online conformal (ACI) + recommendation | **Not found** for graph recommendation. Static conformal recommendation exists (Angelopoulos et al., COPA 2023). |
| Task-free continual graph learning under continuous drift | DRIFT benchmark (arXiv 2605.12998, 2026). Node classification only, but it **shows that continual graph learning methods depend on task boundaries**. This supports our block-free drift trigger. |

**Honest conclusion.** Every individual ingredient exists. The literature does **not** already contain a graph recommender that does all of the following together:

* uses a single probabilistic state, the node posterior, both as its uncertainty estimate and as its anti-forgetting regulariser;
* adapts that state's forgetting rate per node from a calibrated drift test;
* keeps an environment-invariant preference factor separate from a forecastable environment factor;
* initialises new nodes as high-variance priors.

The contribution is therefore an **integration with a specific, testable coupling**, not a new primitive. `research_gap_and_novelty.md` assesses whether that is publishable.
