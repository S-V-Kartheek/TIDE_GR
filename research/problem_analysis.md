# Stage 1 — Problem Analysis

> Status: research design document. No implementation. Claims about specific papers are deferred to `literature_review.md`; this document fixes the *problem* precisely so that later choices can be checked against it.

## 1. The professor's gap, restated

The original gap says that graph recommenders are trained on a historical interaction graph under an implicit IID assumption, so they can be **confidently wrong** when the data-generating process moves. Robustness methods exist (DRO, causal/invariant learning, diffusion denoising), but each targets one facet. Four facets are said to be jointly unaddressed:

1. continuous distribution shift,
2. uncertainty in graph structure and in predictions,
3. catastrophic forgetting under adaptation,
4. open-world (new) users and items.

The fifth facet in the central question — **stable preference vs. spurious temporal correlation** — is the mechanism that is supposed to make generalisation possible.

## 2. Refined central research question

> **RQ.** In a recommendation environment observed as a time-ordered stream of user–item interactions whose distribution drifts and whose node set grows, how can a graph-based recommender (i) maintain a representation of user preference that transfers across temporal environments better than one fitted to the most recent or the pooled history, (ii) quantify — and calibrate — how much each recommendation should be trusted, separating lack of evidence from genuine change, (iii) use that uncertainty to decide *what* and *when* to update, so that adaptation to the present does not erase still-valid past knowledge, and (iv) admit previously unseen users and items without retraining from scratch?

This keeps the professor's objective but makes three things explicit that the original wording left implicit:

* **"Stable" is defined relative to temporal environments**, not as a causal claim (see §5).
* **Uncertainty is an actuator, not only a report**: it drives adaptation and memory decisions. This coupling is where we expect the contribution to sit (validated in `research_gap_and_novelty.md`).
* **Evaluation is prequential (test-then-train)** over future time blocks, not a random split.

## 3. Formal setting (informal version; full math in `mathematical_formulation.md`)

* Users $\mathcal{U}_t$ and items $\mathcal{I}_t$ form **growing** sets; $\mathcal{U}_t \subseteq \mathcal{U}_{t+1}$.
* Interactions arrive as events $(u, i, \tau)$, aggregated into time blocks $B_1, B_2, \dots, B_T$ (e.g., monthly). Block $B_k$ induces bipartite graph $G_k$; the cumulative graph is $G_{\le k}$.
* Each block is drawn from $P_k(u, i)$. **Non-stationarity**: $P_k \neq P_{k+1}$ in general.
* At each step $k$ the model is evaluated on $B_{k+1}$ (future) *before* it sees it, then updated.

## 4. Taxonomy of shift — and which we target

Decompose the joint $P_k(u,i) = P_k(u)\,P_k(i \mid u)$ and further $P_k(i\mid u) \propto f(\text{pref}_u, \text{attr}_i)\cdot g_k(\text{exposure/popularity}_i)$ (a standard multiplicative popularity-exposure factorisation, used e.g. by popularity-deconfounding work; cited in the literature review).

| Concept | Definition in our setting | Targeted? | Why |
|---|---|---|---|
| **OOD generalisation** | Train on $\{P_1..P_k\}$, perform well on unseen $P_{k+1}$ *without* using its data | **Yes (primary)** | This is the "don't be confidently wrong on day one of a new period" requirement. |
| **OOD adaptation / test-time adaptation** | Use a small amount of $P_{k+1}$ data (or unlabeled signals) to adapt | **Yes** | Our continual update step is exactly this. |
| **Online learning** | Per-event updates, regret-oriented | Partially | We do block-incremental updates; per-event streaming is out of scope for compute reasons (see §7). |
| **Continual learning** | Sequence of tasks; must avoid forgetting old tasks | **Yes** | Old periods recur (seasonality), and long-term preferences persist. |
| **Temporal distribution shift** | Umbrella for $P_k$ varying with $k$ | **Yes** | The setting. |
| **Concept drift** | $P_k(i \mid u)$ changes — the preference/exposure mechanism moves | **Yes (core)** | Preference drift + popularity drift. |
| **Covariate shift** | $P_k(u)$ changes (who is active), $P(i\mid u)$ fixed | **Yes (secondary)** | Arises naturally from user-population turnover. |
| **Structural graph shift** | Degree distribution, density, community structure of $G_k$ change | **Yes** | Directly alters GNN message passing; motivates structural uncertainty. |
| **Popularity shift** | $g_k(\cdot)$ changes | **Yes (core)** | The dominant, measurable spurious factor in recommendation. |
| **Preference shift** | $\text{pref}_u$ changes for an individual | **Yes, but must be separated from noise** | The key identification difficulty: is a user's new behaviour a real change or sparse noise? |
| Label/feedback-loop shift | Distribution changes *because* of the recommender's own policy | **No** | Needs interactive/online A/B data; offline public logs cannot test it. Explicitly a limitation. |
| Adversarial shift (attacks) | Injected fake edges | **No (only as perturbation test)** | Separate literature (robust recsys); we use synthetic edge noise only to test structural uncertainty. |

**Decision.** Our target is **gradual and abrupt temporal concept drift in $P(i\mid u)$ driven by popularity/exposure change and preference change, plus covariate and structural shift from node growth**, evaluated as *temporal OOD generalisation followed by block-wise adaptation*.

## 5. What "stable preference" can and cannot mean

From observational logs alone, a "stable preference" cannot be identified causally without assumptions (e.g., known environments with sufficient diversity, a correct causal graph, or exposure data). We therefore adopt an **operational definition**:

> A representation $z_u$ is *temporally invariant* w.r.t. environments $\{e\}$ if the risk of a predictor using $z_u$ has low variance (and low worst case) across environments, and its predictive relationship to interactions $P(y \mid z_u, z_i, e)$ does not depend on $e$.

This is **statistical invariance**, which is what IRM/REx/GroupDRO-style objectives give. It coincides with causal invariance only under the assumptions those papers state (discussed in `architecture_comparison.md`). We will claim the weaker property and test it empirically (cross-period transfer, popularity-shift tests), not claim causal discovery.

Two kinds of spurious temporal signal are distinguished:

* **Observable confounders** (item popularity in a period, item age, seasonal category mass) — can be modelled explicitly as an environment-specific branch.
* **Latent short-lived factors** (ad campaigns, news events) — only detectable via environment inference or via residual instability; no method guarantees removal.

## 6. Uncertainty — what must be separated

| Quantity | Meaning here | Source |
|---|---|---|
| Edge (structural) uncertainty | Is edge $(u,i)$ informative of preference, or noise/accidental/stale? | Implicit feedback is noisy; old edges become stale under drift |
| User/item embedding uncertainty | How well-determined is $z_u$ / $z_i$ given the evidence? | Low degree, new nodes, recent change |
| Predictive uncertainty | Spread of the score $s(u,i)$ or of $P(y=1\mid u,i)$ | Combination of the above + model uncertainty |
| Epistemic | Reducible with more data — **lack of evidence** | Cold nodes, novel regions of embedding space |
| Aleatoric | Irreducible — **inherent unpredictability** of the user's choices | Exploratory users, broad tastes |
| Calibration | Do confidence values match empirical frequencies? | Requires a probabilistic output and held-out future data |

The research question demands telling **"insufficient evidence"** (epistemic) apart from **"preferences changed"** (a shift). These are confounded in a single variance number; a model is needed in which (a) epistemic uncertainty decreases with evidence, (b) drift increases a separate *mismatch* signal (predicted vs. realised), and (c) calibration is maintained under drift.

## 7. Constraints that shape the design

* **Final-year project compute**: single or few GPUs; datasets of 10^5–10^7 interactions. Deep ensembles of large GNNs over many time blocks, or per-event continuous-time GNNs over 10^7 events, are borderline.
* **Scientific clarity**: every module must be removable in an ablation and must correspond to one hypothesis.
* **Distinction from MM-CLightRec**: the contribution must be in the *temporal/continual/uncertainty coupling*, not in multimodal fusion.

## 8. Relationship to MM-CLightRec (prior work of the student)

Based only on the architectural description provided (no code inspected):

| MM-CLightRec component | Keep? | Reasoning |
|---|---|---|
| LightGCN channels | **Keep the propagation rule as the backbone (single channel)** | Simple, linear, well-understood; its linearity makes uncertainty propagation and Laplace/Gaussian treatments tractable. Not sufficient for time by itself — time must enter through environments, edge weighting, and incremental updates, not through the encoder. |
| Visual/textual/categorical channels + adaptive fusion + cross-attention | **Discard as contribution; keep only a single lightweight text/category feature encoder for cold-start** | Multimodality is orthogonal to the RQ; keeping it would confound attribution and invite "MM-CLightRec + time" criticism. Item side-information is needed *only* for open-world items; one frozen text encoder suffices. |
| Hierarchical contrastive learning | **Discard** | Contrastive augmentation is not an invariance mechanism across *time*; it would add a loss whose purpose overlaps with the invariance objective. |
| VGAE uncertainty | **Replace** | VGAE gives a posterior over node embeddings for reconstruction; it does not separate edge from prediction uncertainty, is known to be poorly calibrated without care, and its reparameterised variance is not updated coherently over time. We replace it with an explicit edge-reliability posterior + Gaussian embedding posterior with a principled temporal update (justified in `architecture_comparison.md`). |
| Cold-start module | **Re-design** | Must be inductive and uncertainty-aware (new nodes start with high epistemic variance and a prior mean from features/neighbours). |

Net: the new project re-uses **only** the LightGCN propagation rule and the idea of a feature-based cold-start prior. Everything that constitutes the contribution is new.
