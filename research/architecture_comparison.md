# Stage 4 — Candidate Architecture Comparison

For each challenge this document lists:

* the candidates and their assumptions;
* the evidence, with sources in `literature_review.md`;
* compatibility with the other modules;
* cost;
* the decision and why the alternatives were rejected.

The guiding rule is to choose **the simplest mechanism that is scientifically adequate and that participates in the coupling that forms our contribution**.

---

## 1. Graph encoder

| Candidate | Assumption / mechanism | Evidence | Cost per block | Compatibility | Verdict |
|---|---|---|---|---|---|
| **LightGCN** (He et al., SIGIR'20) | Linear propagation of ID embeddings over the normalised bipartite graph | Strong, reproducible CF baseline; the backbone of DR-GNN, InvCF, DRGO, CausalDiffRec, GraphPro and DEGC | O(L·\|E\|·d) | **Linear in node parameters and in edge weights**, so mean and variance propagate in closed form, edge weights enter directly, and the Laplace curvature is cheap | **Selected as backbone** |
| NGCF (SIGIR'19) | Non-linear propagation with feature transforms | LightGCN outperforms it with fewer parameters (LightGCN paper) | Higher | Non-linearity breaks closed-form uncertainty propagation | Rejected |
| SimGCL / XSimGCL / SGL / LightGCL | LightGCN + contrastive augmentation | Higher static accuracy | +30–100% | Orthogonal; could be added later. Its contrastive objective does not target temporal invariance and would complicate ablation | Not in core; optional add-on noted as future work |
| TGN / TGAT / JODIE (event-based, continuous-time) | Per-node memory updated per event; temporal attention | Strong on dynamic link prediction. **But** EdgeBank (NeurIPS'22 D&B) shows memorisation is competitive under lenient negatives; GraphMixer (ICLR'23) shows simple MLPs rival complex temporal models; TGB (NeurIPS'23) finds simple methods often win on node tasks; TGSRec's repository reports bugs and reruns | O(#events) sequential memory updates; hard to batch over 10^6–10^7 events × many blocks | Memory states are deterministic, so they would need a separate uncertainty treatment; per-event updates conflict with block-level calibration and invariance environments | **Rejected as backbone** (cost, weak evidence for top-K rec, poor fit with Bayesian state) |
| DyGFormer / graph transformers | Attention over neighbour sequences | SOTA on DyGLib; some baselines are irreproducible | High | Same issues as TGN | Rejected |
| Snapshot-RNN (EvolveGCN, ROLAND) | GNN weights evolve via an RNN across snapshots | Used as baselines in GraphPro, where GraphPro beat them | Moderate | Deterministic weight evolution; does not give per-node uncertainty | Rejected |

**Snapshot vs. event vs. hybrid.** We choose a **hybrid**:

* the *encoder* is snapshot-based, run on the cumulative graph at block boundaries;
* *time enters* through three routes: (i) edge-reliability priors that decay with edge age, (ii) per-node state dynamics (process noise) between blocks, and (iii) environment features such as within-block popularity and its trend.

This keeps the encoder simple and attributes temporal gains to the mechanisms under study rather than to encoder capacity.

**Is LightGCN sufficient for temporal requirements?** Not alone. Static LightGCN is the "IID" baseline the professor's gap criticises. It becomes adequate only together with modules M2, M3 and M5 below. We test this directly: "LightGCN retrained per block" and "LightGCN fine-tuned per block" are baselines.

---

## 2. Stable preference learning (Challenge B)

| Candidate | What it actually guarantees | Evidence and caveats | Fit to our setting | Verdict |
|---|---|---|---|---|
| **IRM / IRMv1** (Arjovsky'19) | Invariant predictor across *given* environments, in the linear case with enough environments | Fails in non-linear regimes or with few environments (Rosenfeld et al., ICLR'21); unstable penalty; ERM is competitive under fair tuning (Gulrajani & Lopez-Paz, ICLR'21) | The gradient penalty on a dummy classifier is awkward with pairwise ranking losses | Rejected |
| **V-REx** (Krueger et al., ICML'21) | Penalises variance of risks across environments; extrapolates between ERM and worst-case robustness | Simple; works with any loss; same environment-diversity caveat as IRM | **Works directly on per-block BPR risks**; environments = time blocks (current + replayed) | **Selected** |
| GroupDRO (Sagawa et al., ICLR'20) / KL-DRO / Sinkhorn-DRO (DR-GNN, DRGO) | Minimises the worst-case environment or group risk within an uncertainty set | Robust to the worst group but can be pessimistic; DR-GNN's gains are weak on temporal splits | Redundant with V-REx (V-REx with β→∞ approximates worst-case); DRO adds an adversary inside an already-coupled optimisation | Rejected for the core; **DR-GNN and DRGO kept as baselines** |
| Environment inference (EIIL, Creager'21; InvPref; CausalDiffRec) | Learns latent environment partitions when labels are missing | Useful when environments are unknown | We *have* natural environments (time blocks). Inferring extra partitions doubles the optimisation | Not in core; **ablation**: replace time-block environments with EIIL-inferred ones |
| Disentangled preference/popularity (DICE, InvCF, PDA) | Separates conformity or popularity from interest. PDA: deconfounded training plus forecast popularity at inference | PDA: large gains under popularity drift (MF). InvCF: popularity-shift robustness | **Adopted as the structural form of the score**: the invariant factor plus an explicit environment term built from *observable* popularity and recency statistics | **Selected (score factorisation)** |
| Causal representation learning (COR, CausPref, CausalDiffRec, CILER) | Identification under a stated causal graph or exponential-family assumptions | Requires assumptions we cannot test on logs | We do not claim causal identification | Rejected as a claim; cited as related work |
| Contrastive invariance (AdvDrop, SimGCL-style) | Representation invariance to augmentations | Augmentation invariance ≠ temporal invariance; "Does Invariant Graph Learning via Environment Augmentation Learn Invariance?" (NeurIPS'23) casts doubt | Overlaps with V-REx | Rejected |

**Which mechanisms identify stable preferences, and which only encourage statistical invariance?**

* *Identification* requires either interventions (randomised exposure, as in KuaiRand's random slice) or a correct causal model plus sufficient environment heterogeneity.
* Everything else, including V-REx, IRM, DRO, contrastive and disentangled methods, enforces **statistical invariance across the observed environments**.

Our design therefore makes a narrower, defensible claim:

* The preference factor is penalised for **risk instability across time blocks** after conditioning on **observable** environment covariates.
* We test the claim on future blocks and popularity-shift subsets.
* It is not a causal claim.

---

## 3. Shift detection

| Candidate | Signal | Problem in our setting | Verdict |
|---|---|---|---|
| MMD (two-sample kernel test) on embeddings | Distribution of node embeddings across blocks | The embedding coordinate system changes with each update, so the test conflates model change with data change. It is not localised to nodes. Quadratic cost | Rejected as the primary detector |
| Wasserstein distance on embeddings | Same | Same issues; sliced versions are cheaper but still global | Rejected |
| Graph topology monitoring (degree distribution, density) | Structural shift | Cheap and model-free, but says nothing about *preference* change | **Kept as a global descriptive monitor** (reported, not used for gating) |
| Popularity-distribution divergence (JS between block item-popularity distributions; used by PDA) | Popularity/environment shift | Model-free, interpretable | **Selected (global environment monitor)**; drives the environment-term refit |
| Sequential drift detectors on loss (ADWIN, Page–Hinkley, CUSUM) | Change in a scalar loss stream | Classic and well understood. On raw loss they confuse sparsity with drift: low-degree nodes have noisy losses | **Selected in a modified form**: CUSUM applied to a *standardised predictive surprise* (below) |
| **Standardised predictive surprise** (ours; a calibration-based z-statistic) | Realised next-block NLL of node *v* minus its expectation under the model's own predictive distribution, divided by the predictive standard deviation | Requires approximate calibration (handled by M6). Its null distribution under "no change" is approximately N(0,1) by a CLT over interactions | **Selected (per-node detector)**. It is the mechanism for TQ2: high epistemic variance inflates the denominator, so sparse users are *not* flagged, while confident users who change *are* flagged |
| Label-free proxies (LiDu "Are Recommenders Self-Aware?", 2025 preprint) | List-level uncertainty as a performance proxy | Promising but unreviewed | Cited; not used |

**Trigger logic.**

* Adaptation is *always* incremental per block, because blocking is cheap.
* Drift scores **modulate how much** each node may move, through process noise, and **what is replayed**.
* A global alarm (popularity JS or mean surprise above a CUSUM threshold) additionally triggers:
  * re-fitting the environment-term forecaster;
  * an extended consolidation epoch budget.

This avoids the brittle binary "retrain / don't retrain" decision, and the DRIFT benchmark (2026) shows that continual graph learning methods suffer when they rely on task boundaries.

---

## 4. Uncertainty (Challenge C)

### 4.1 Structural (edge) uncertainty

| Candidate | Mechanism | Assessment | Verdict |
|---|---|---|---|
| VGAE (Kipf & Welling'16) | Gaussian latent per node; reconstructs edges | The uncertainty is over *node latents for reconstruction*, not over the reliability of *observed* edges. No temporal prior. Typically uncalibrated | **Rejected** (this is the MM-CLightRec component; replaced) |
| Bayesian GCN random-graph posterior (Zhang et al. AAAI'19; BGCN-Rec KDD'20) | Observed graph as one sample from a parametric random graph (MMSBM / node copying) | Principled, but the MMSBM fit is expensive and static | Inspiration |
| Deterministic denoising (RGCF SIGIR'22, T-CE WSDM'21, DR-GNN reweighting) | Learned or loss-based edge weights | No posterior. High-loss edges are treated as noise, which is wrong under drift | Baselines and ablation |
| **Variational edge-reliability posterior** (ours): Bernoulli latent $r_{ui}$, amortised $q(r_{ui})=\mathrm{Bern}(\pi_{ui})$, **time-decaying prior** $\rho(\Delta t)$, Concrete relaxation | Gives a probability that each edge is preference-consistent; staleness enters through the prior, not the loss | Cheap (an MLP over edge features); entropy is interpretable; its effect propagates into node variance | **Selected** |
| Diffusion denoising (DRGO, CausalDiffRec) | Denoises the latent space | Expensive per update; not a calibrated edge probability | Baseline only |

### 4.2 Node (embedding) and predictive uncertainty

| Candidate | Epistemic quality | Under shift | Cost in a continual loop | Coupling with forgetting / cold-start | Verdict |
|---|---|---|---|---|---|
| Deep ensembles (M=5) | Best empirically under shift (Ovadia et al.'19) | Good | M× training **per block** and M× memory | None natural: five independent models do not give a regulariser | **Upper-reference only** (small dataset) |
| MC dropout | Weak; underestimates | Degrades | Cheap | None | Rejected |
| Evidential (GPN, GKDE, E-NSDE) | Contested: second-order loss minimisation does not give faithful epistemic uncertainty (Bengs et al., NeurIPS'22; Meinert et al., AAAI'23) | Mixed | Cheap | None | Rejected |
| Full variational BNN (VCL) | Principled | Good with forgetting (Kurle'20) | 2× parameters plus MC sampling; noisy gradients on 10^6 node embeddings | Natural | Rejected for cost and noise |
| **Diagonal Laplace / Gaussian node posterior with online precision accumulation and Bayesian forgetting** (Ritter'18; Daxberger'21; Kurle'20) | Good for its cost (Laplace Redux finds it competitive) | Explicit process noise models drift | **Closed-form GGN diagonal for BPR on LightGCN** (derived in `mathematical_formulation.md`): one extra pass | **The precision is simultaneously** (a) epistemic uncertainty, (b) the online-Laplace (EWC-generalising) anti-forgetting weight, (c) the cold-start prior | **Selected** |
| Conformal (split, ACI, Conformal PID) | Distribution-free coverage; not a model of uncertainty | **ACI guarantees long-run coverage under arbitrary shift** (Gibbs & Candès'21) | Negligible | Wraps any score | **Selected as the calibration/selective layer**, not as the uncertainty model |

**Are separate mechanisms needed for structural and predictive uncertainty?** Yes, but they must *compose*:

* edge uncertainty, $\pi(1-\pi)$, and node uncertainty, $\sigma^2$, are separate latent sources;
* both propagate through the same linear encoder into one predictive variance;
* that variance is decomposed into $v^{\text{node}}$ and $v^{\text{edge}}$ components.

A single mechanism such as VGAE cannot tell "this user's edges are unreliable" apart from "this user has few edges".

**Aleatoric vs. epistemic.**

* Epistemic = variance of the score induced by the posterior over parameters and edges.
* Aleatoric = expected Bernoulli entropy $\mathbb{E}[p(1-p)]$ at the predictive mean.

This is the standard total-variance / mutual-information decomposition (Kendall & Gal'17; Depeweg'18), applied to our predictive Gaussian.

---

## 5. Continual learning (Challenge D)

| Candidate | Memory | Compute | Evidence in graph rec | Interaction with OOD / uncertainty | Verdict |
|---|---|---|---|---|---|
| Fine-tuning | 0 | 1× | Forgets (all CL papers) | — | Baseline |
| Full retrain on all data | All data | Grows with time | Upper bound on retention, but stale for adaptation | — | Baseline (reference) |
| EWC (PNAS'17) | O(\|θ\|) | ~1× | Used as a baseline in graph CL | A global λ; importance is not updated under drift | Subsumed (our online Laplace with forgetting generalises it) |
| **Online Laplace with Bayesian forgetting** (Ritter'18; Kurle'20) | O(\|θ\|) | ~1× | Not yet applied to graph rec (gap) | **Per-node precision acts as importance; per-node process noise acts as plasticity gated by the drift test** | **Selected** |
| Experience replay (Chaudhry'19; SAER CIKM'21) | Buffer \|M\| | +\|M\| | Strong and simple | **Required anyway**: the V-REx penalty needs past environments | **Selected**, with environment-stratified, reliability- and drift-aware selection |
| Knowledge distillation (GraphSAIL, LWC-KD, LwF) | Old model | +1 forward pass | Standard in graph rec | Distils old predictions *including stale ones*, so it conflicts with adaptation unless gated (PIW gates it heuristically) | Rejected (redundant with Laplace + replay); **GraphSAIL / LWC-KD as baselines** |
| Model isolation / expansion (DEGC) | Grows | Moderate | Strong on Taobao | No uncertainty semantics | Baseline |
| Prompt-based (GPT4Rec, GraphPro, GPL4SRec) | Small prompts | Low | Current SOTA family in streaming graph rec | A frozen backbone limits invariance learning | **Baselines** |
| Parameter-efficient (LoRA-style adapters) | Low | Low | Mostly LLM rec | Could replace the Laplace regulariser; does not give uncertainty | Rejected |

---

## 6. Open-world (Challenge E)

| Candidate | Mechanism | Assessment | Verdict |
|---|---|---|---|
| Retrain with new IDs | — | Expensive; violates the requirement | Rejected |
| GraphSAGE / PinSage inductive encoders | Feature-based aggregation | Requires rich node features; users have none in Amazon | Partially (items only) |
| IDCF (ICML'21) / INMO (SIGIR'22) | Represent new users through key/template users or items they interacted with | Strong inductive CF; deterministic | **Adopted as the prior mean** for new users with ≥1 interaction |
| Feature regression (DropoutNet, CLCRec, ALDI, Heater) | Map item content to the CF space | Standard item cold-start; TDRO adds temporal robustness | **Adopted as the prior mean** for new items: a precision-weighted regression of warm item means from text embeddings |
| Meta-learning (MeLU, MetaHIN) | Fast adaptation from few shots | Bi-level optimisation per block is expensive; no calibrated variance | Rejected |
| LLM simulators (2024–2026) | Synthesise interactions | Contamination risk; cost; off-topic | Rejected |
| **Bayesian new-node prior** (ours) | Prior mean from the above; **prior precision low and evidence-dependent**; updated in closed form as interactions arrive | Integrates with uncertainty-aware ranking: new nodes are automatically discounted and flagged; no retraining | **Selected** |

**Are multimodal representations needed?** No. Only *item text* matters, and only to set a prior mean for new items. A single frozen text encoder and a linear map suffice. Visual channels, cross-attention and adaptive fusion (from MM-CLightRec) are **excluded**: they add capacity unrelated to the research question and confound attribution.

---

## 7. Assembling candidates: three coherent systems compared

| | **S1 (selected): TIDE-GR** | **S2: Event-based ensemble** | **S3: Streaming robust-prompt** |
|---|---|---|---|
| Encoder | Edge-weighted LightGCN, block snapshots | TGN-style memory + temporal attention | LightGCN backbone + GPT4Rec/GraphPro prompts |
| Invariance | V-REx over time blocks + observable environment term | DRO over time windows | DR-GNN KL-DRO inside each update |
| Structural uncertainty | Variational edge reliability with temporal prior | None (or MC edge dropout) | DR reweighting (worst case) |
| Predictive uncertainty | Gaussian node posterior (online Laplace) → closed-form score variance; ACI calibration | Deep ensemble (M=5) + temperature scaling | None, or MC dropout |
| Continual | Precision-weighted regulariser + drift-gated forgetting + environment-stratified replay | Replay + EWC | Frozen backbone + prompts |
| Open-world | Bayesian prior from IDCF/INMO neighbour mean or text regression | Memory initialised at zero (TGN) | Prompt edges |
| Cost/block | ≈1.3–1.6× LightGCN fine-tune (estimate; must be measured) | ≈5× TGN training (ensemble) | ≈1.1× |
| Coupling | **One state drives uncertainty, retention, adaptation and cold-start** | Modules independent | Modules independent |
| Main risk | Laplace epistemic estimates cruder than ensembles; invariance gain may be small | Compute; weak top-K evidence for event-based encoders | Little research novelty (it is a combination of baselines) |

**Strongest alternative: S2 (event-based encoder + deep ensemble + replay/EWC + DRO).** It was not selected because:

1. M× per-block training is impractical over about 27 blocks on a student budget.
2. Evidence that complex continuous-time encoders beat simple ones for *top-K recommendation* is weak (EdgeBank, GraphMixer, TGB).
3. Ensembles give no natural anti-forgetting regulariser or cold-start prior, so the modules stay uncoupled and the contribution reduces to engineering.

**When S2 would win:**

* sub-session dynamics matter, for example click streams with minute-level intent (Taobao, KuaiRand), which block snapshots cannot represent;
* compute is abundant and uncertainty quality is the primary metric, since ensembles are the most robust under shift (Ovadia'19).

We include a deep ensemble of *our own backbone* as an uncertainty upper-reference to quantify this gap.
