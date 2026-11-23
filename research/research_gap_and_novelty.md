# Stage 3 — Research Gap Validation and Novelty

> Convention:
>
> * **[V]** = verified finding, with its source in `literature_review.md` and `closest_competing_methods.md`.
> * **[H]** = our hypothesis or judgement.
> * **[R]** = a risk.

## 1. Original gap (the professor's paragraph)

> Graph recommenders assume IID historical data. Robustness methods (DRO, causal, diffusion, invariant) do not *jointly* address continuous shift, structural and predictive uncertainty, catastrophic forgetting and open-world nodes. As a result, models are confidently wrong after preferences or item distributions change.

## 2. What the literature shows

### 2.1 Already solved, or substantially addressed

| Problem | Evidence |
|---|---|
| Static OOD robustness of graph CF to popularity, exposure and one-shot temporal shift | [V] DR-GNN (WWW'24), DRGO (WWW'25), CausalDiffRec (WWW'25), InvCF (WWW'23), AdvDrop (WWW'24) |
| Popularity–preference disentanglement and popularity-drift forecasting | [V] PDA (SIGIR'21), InvCF, DICE (WWW'21) |
| Incremental / streaming graph recommendation with less forgetting | [V] GraphSAIL (CIKM'20), SAER and LWC-KD (CIKM'21), PIW (AAAI'23), DEGC (WWW'23), GPT4Rec (SIGIR'24), GPL4SRec (IJCAI'25), GraphPro (WWW'24) |
| Temporal + distributionally robust cold-start for items | [V] TDRO (AAAI'24) |
| Inductive representations for unseen users and items | [V] IDCF (ICML'21), INMO (SIGIR'22), and feature-based cold-start (DropoutNet, CLCRec, ALDI) |
| OOD generalisation on generic dynamic graphs with evolving environments | [V] DIDA (NeurIPS'22), EAGLE (NeurIPS'23), EvoOOD (TPAMI, accepted) |
| Static calibration of ranking scores | [V] Kweon et al. (AAAI'22); PerK (WWW'24) |
| Distribution-free reliability for recommendation, assuming exchangeability | [V] Angelopoulos et al. (COPA'23) |
| Online calibration with coverage guarantees under arbitrary shift (generic) | [V] ACI (NeurIPS'21); Conformal PID (NeurIPS'23) |
| Bayesian continual learning with adaptive forgetting (generic) | [V] VCL (ICLR'18), online structured Laplace (NeurIPS'18), Kurle et al. (ICLR'20) |
| Random-graph posterior for GNN recommendation (static) | [V] BGCN-Rec (KDD'20) |

### 2.2 Partially solved

* **Temporal shift in graph OOD recommendation** [V]. DR-GNN, DRGO, CausalDiffRec and InvGCLLM treat time as one train/test split (ML-1M, Food, Douban). None updates over time, and DR-GNN's gains on temporal splits are small: Food +3% NDCG@20 over LightGCN, and below APDA on ML-1M.
* **Stable vs. drifting users in continual graph recommendation** [V]. PIW and DEGC separate them with learned heuristics (an imitation weight, pruning). Neither gives a statistical test or an uncertainty interpretation.
* **Uncertainty in graph recommendation** [V]. BGCN-Rec, UCC (SIGIR'23) and W-GAT use uncertainty for accuracy, diversity or cold-start. None evaluates calibration, and none works under temporal shift.
* **Temporal generalisation in incremental industrial pipelines** [V]. ELBO_TDS (preprint, 2025) is not graph-based and reports no calibration or forgetting metrics.

### 2.3 Not adequately solved

Three independent search passes found no verified paper that does any of the following:

1. Evaluates graph recommenders **prequentially** across many future blocks for *both* OOD generalisation (before adaptation) *and* forgetting (after adaptation).
2. Provides **calibrated** predictive uncertainty for a graph recommender **under temporal shift**, or in a streaming setting.
3. Uses uncertainty to **decide how much each node's representation should change** when new data arrive in a recommender.
4. Separates "**insufficient evidence**" from "**preference changed**" with an explicit, testable criterion.
5. Combines environment-invariant preference learning with continual updating, where the invariance penalty must be computed over environments that are no longer in the training window.

Search caveat: this is absence of evidence from searches on 2026-11-21. Industrial work or non-indexed workshop papers may exist. [R]

## 3. Limiting assumptions in current methods

1. **A single fixed test distribution.** Static OOD rec assumes one shift between train and test. In practice the shift is a sequence.
2. **Known task boundaries and stationarity within a task.** Continual graph rec uses fixed segments. DRIFT (2026) shows that continual graph learning methods degrade when boundaries are removed [V].
3. **Exchangeability for calibration and conformal guarantees.** Kweon'22 and Angelopoulos'23 assume it, and drift violates it [V].
4. **Uncertainty ≠ change.** Methods that down-weight high-loss interactions (T-CE/ADT, RGCF) treat surprising edges as noise. Under drift, surprising edges are often *signal*. [V for the methods; H for the consequence.]
5. **Environment labels or diversity.** IRM-family guarantees require enough heterogeneous training environments [V: Arjovsky'19; Rosenfeld'21]. In streaming rec, only the current window is available unless memory is kept.

## 4. Answers to the ten guiding questions

1. **What is solved?** See §2.1. Static OOD graph rec, streaming graph rec without OOD, temporal cold-start without graphs, and generic online calibration.
2. **What is partially solved?** See §2.2.
3. **What is not adequately solved?** See §2.3.
4. **Which assumptions limit current methods?** See §3.
5. **Why is continuous adaptation difficult?**
   * The model must decide, per user, whether a deviation is noise, a transient environment effect, or a durable preference change.
   * The data for that decision are sparse: most users have few interactions per block.
   * Retraining frequency is constrained by compute (the Continual RecSys tutorial, 2025 [V]).
   * The model's own embedding coordinates move, so naive distribution monitoring in embedding space is ill-posed [H].
6. **Why can invariant preference learning fail?**
   * Invariance objectives only remove variation that is *present across training environments*. A novel environment, such as a new category or a new kind of campaign, has no counterpart.
   * IRMv1 can fail in the non-linear regime, and with too few environments [V: Rosenfeld et al., ICLR'21].
   * Under fair tuning, ERM is often as good as domain-generalisation methods [V: Gulrajani & Lopez-Paz, ICLR'21; Wild-Time, NeurIPS'22 found no method consistently better than ERM].
   * Environment augmentation may not yield invariance [V: Chen et al., NeurIPS'23].
   * Preferences themselves change. An invariance constraint can then wrongly freeze a real change [H].
7. **Why is uncertainty difficult in evolving graphs?**
   * The graph is both input and label, so edge noise affects both message passing and supervision.
   * Node evidence is heterogeneous: degrees are power-law distributed.
   * Calibration targets drift.
   * Edge-level and node-level uncertainty interact through propagation.
   * Ensembles are the most robust under shift [V: Ovadia et al.'19], but cost M× per block in a continual loop.
   * Evidential methods are cheap but their epistemic estimates are contested [V: Bengs et al.'22; Meinert et al.'23].
8. **Why can continual learning conflict with OOD adaptation?**
   * Regularisation and replay keep the model near past solutions, while adaptation requires moving away.
   * A global trade-off knob (the EWC λ, the replay ratio) is wrong for both drifted and stable users at once.
   * Invariance needs old environments (memory), while forgetting stale ones needs eviction. The memory policy must serve both.
9. **Why is open-world adaptation difficult in graph recommendation?**
   * Transductive ID embeddings have no parameters for new nodes.
   * New nodes have few edges, so message passing is unreliable.
   * Feature-based priors can be biased toward the training-period distribution (TDRO's motivation [V]).
   * New nodes are exactly where confident errors are most likely, yet most cold-start methods output point estimates [H].
10. **What is the precise opportunity?** See §5.

## 5. Refined research gap

> **Refined gap.** Graph recommenders that are robust to distribution shift are trained and evaluated against a single, static shift. Graph recommenders that learn continually from streams do not generalise out of distribution and do not know when they are wrong. No existing graph recommender maintains a **probabilistic belief over user and item representations** in which the same quantity:
>
> * measures how reliable a recommendation is;
> * sets how strongly past knowledge is retained;
> * sets how quickly a node may change, through a calibrated drift test;
> * initialises unseen nodes.
>
> Without such a belief, three things go wrong. Stability and plasticity are traded off globally rather than per node. "Unknown" and "changed" are conflated. Reliability claims are invalid as soon as the distribution moves.

## 6. Technical research gap (formal questions)

* **TQ1.** Can a Gaussian node posterior with a per-node process-noise (forgetting) term, of Kalman / Bayesian-forgetting type, serve simultaneously as the epistemic uncertainty and as the continual-learning regulariser of a LightGCN-type recommender, without degrading accuracy?
* **TQ2.** Does a standardised predictive-surprise statistic, which compares realised next-block likelihood with the model's own predictive expectation, discriminate genuine preference change from data sparsity better than raw loss, raw variance or embedding distance?
* **TQ3.** If the score is factorised into an environment-invariant preference term plus a forecastable environment (popularity/recency) term, and a risk-variance penalty is applied across time blocks held partly in replay memory, does generalisation to the *next* block improve over ERM, DRO and existing invariant graph CF?
* **TQ4.** Does an edge-reliability posterior with a time-decaying prior identify stale or noisy edges, and is its contribution to predictive variance informative beyond node uncertainty?
* **TQ5.** Can calibration be maintained across blocks with an online adaptive conformal update, and does uncertainty-aware (selective) ranking reduce confident errors on new and drifted nodes?
* **TQ6.** Can new nodes be admitted at O(degree) cost with a feature- or neighbour-based prior mean and a high prior variance, matching or exceeding inductive and cold-start baselines while being better calibrated?

## 7. Proposed contribution (what we will claim)

* **C1 — Problem formulation and protocol.** A prequential *temporal-OOD continual recommendation* protocol that measures, on the same blocks: (i) next-block accuracy before adaptation (OOD), (ii) retained accuracy on past blocks (forgetting / BWT), (iii) calibration and selective risk under drift, and (iv) cold-node performance against evidence. No existing graph-rec benchmark reports all four [V by absence; R].
* **C2 — Method.** TIDE-GR, specified in `final_architecture.md`:
  * a single **drift-gated Bayesian node state** couples uncertainty, retention and adaptation;
  * an **invariant/environment score factorisation** with a cross-block risk-variance penalty computed over replay memory;
  * an **edge-reliability posterior**;
  * **online calibrated** selective ranking.
* **C3 — Diagnostic.** A standardised-surprise drift statistic with a stated null distribution, separating epistemic sparsity from change. Tested on synthetic injected preference swaps and real data.
* **C4 — Empirical study.** Including the negative cases: when invariance or uncertainty does *not* help.

What we will **not** claim:

* causal identification of preferences;
* a new GNN encoder;
* a new conformal method;
* a new continual-learning algorithm in isolation.

## 8. Research hypotheses

| ID | Hypothesis | Falsified if |
|---|---|---|
| H1 | TIDE-GR's next-block Recall/NDCG@20, averaged over blocks, is ≥ the best of {DR-GNN, DRGO, GPT4Rec, DEGC, GraphPro} under the same prequential protocol | Significantly worse on ≥2 of 3 datasets |
| H2 | The invariance penalty (λ_inv>0) with the environment term reduces the drop from in-period to next-period accuracy, especially on items whose popularity shifts most | No significant reduction vs. λ_inv=0 under tuned ERM (an honest risk given Gulrajani & Lopez-Paz) |
| H3 | Standardised surprise detects injected preference changes with higher AUROC than raw-loss, raw-variance and embedding-shift detectors, and its false-alarm rate on low-degree users is lower | AUROC is not better, or the false alarms concentrate on sparse users |
| H4 | Drift-gated process noise dominates uniform EWC / online-Laplace and fine-tuning on the stability–plasticity frontier (forgetting vs. next-block accuracy) | The frontier is not dominated |
| H5 | Online calibration keeps next-block ECE and coverage error lower than static post-hoc calibration (Kweon'22, temperature scaling) across blocks | Equal or worse |
| H6 | Selective ranking via predictive variance reduces the error rate at fixed coverage more than score-only confidence, specifically on cold and drifted users | Risk–coverage curves are not better |
| H7 | Edge-reliability entropy ranks synthetically injected noisy or stale edges above genuine ones (AUROC > 0.5 significantly), and removing the module increases degradation under edge noise | Otherwise |
| H8 | New-node priors give better cold-node Recall at 0–5 interactions than INMO/IDCF-style and feature-regression baselines, with lower ECE, at O(degree) insertion cost | Otherwise |

## 9. Novelty risks (honest)

| Risk | Severity | Mitigation |
|---|---|---|
| **R1.** "Integration of known parts." Each primitive (online Laplace, Bayesian forgetting, V-REx, ACI, Concrete edge masks, INMO-style init) exists | **High** | Make the *coupling* the claim (C2/C3), and show by ablation that the coupled version beats the same parts used independently (e.g., EWC + separate MC-dropout + separate drift detector). |
| **R2.** Invariance gains may vanish under fair ERM tuning (well-documented in DG) | Medium–High | Pre-register the tuning budget; report even if negative. The paper survives through C1/C3/H4–H6. |
| **R3.** A concurrent paper on "uncertainty-aware continual graph rec" appears (fast-moving area: e.g., BAKE WWW'26 for KGs, DRIFT 2026) | Medium | Re-run the literature search before submission; keep the protocol contribution (C1). |
| **R4.** Calibration in implicit-feedback rec depends on the negative-sampling definition | Medium | Fix and report the definition (§ in `experimental_design.md`); evaluate full-ranking metrics separately. |
| **R5.** Gaussian / Laplace epistemic estimates are cruder than deep ensembles | Medium | Include an ensemble upper-reference on the small dataset, and report the cost/quality trade-off. |
| **R6.** Standardised surprise assumes approximate calibration; it is miscalibrated early | Medium | Warm-up phase; the ACI-corrected predictive distribution feeds the statistic. |

## 10. Evidence required to substantiate novelty

1. A literature re-check in the month before submission (arXiv cs.IR/cs.LG, SIGIR/KDD/WWW/RecSys 2026–27 proceedings).
2. **Coupling ablation**: TIDE-GR vs. "parts used independently" with equal compute.
3. Results on ≥2 real datasets plus one controlled synthetic-drift variant.
4. All of H1–H8 reported, including failures.
5. Released code and splits, so the C1 protocol can be adopted.
