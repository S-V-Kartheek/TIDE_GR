# Stage 8 — Publication and Novelty Assessment

> No acceptance is guaranteed. This is a preliminary, honest assessment made **before any experiments**. Venue facts come from the official CFP pages, checked on 2026-11-21; NV = not verified.

## 1. Venue comparison

| Venue | Scope fit | Format (latest CFP) | Expectations | Next deadline | Fit for this project |
|---|---|---|---|---|---|
| **SIGIR 2027** | Core IR/RecSys; robust and continual rec papers appear regularly (GPT4Rec '24, DROS '23, RGCF '22) | 9 pp. incl. appendix, references unlimited; double-anonymous (sigir2027.org) | Strong baselines, full-ranking evaluation, ablations, code | Abstract 14 Jan 2027, paper 21 Jan 2027 | **Good, but the timeline is tight.** The deadline is about 2 months away, before experiments exist. Not realistic for the full study |
| **KDD 2027** (Research, cycle 2) | Methods + data mining; DRO/causal rec papers appear | 8 content pp. + unlimited appendix at submission | Rigour, scalability, theory welcome | "Feb 2027" (exact date NV) | Possible stretch target if results are ready by Feb 2027; otherwise KDD 2028 cycle 1 (≈ Jul 2027, NV) |
| **WWW 2027** | Home of DR-GNN, DRGO, CausalDiffRec, InvCF, GraphPro | 8 pp. + refs/appendix ≤12 | Same as SIGIR | **Closed** (25 Oct 2026) | WWW 2028 (≈ Oct 2027, NV): a realistic primary target |
| **WSDM 2027** | Web/data mining | 9 pp. | Solid but slightly lower bar | **Closed** | WSDM 2028 (≈ Aug 2027, NV) |
| **RecSys 2027** | Core recommender venue; values evaluation rigour and reproducibility; has a Reproducibility track | Long: 8 content pp.; long-paper acceptance 20–24% (RecSys'26 CFP) | Evaluation methodology is valued, which suits contribution C1 | 2027 CFP NV (2026: mid-April) | **Very good fit, and a realistic first target** (≈ April 2027) |
| **CIKM 2027** | Broad; many incremental graph rec papers (GraphSAIL, SAER, LWC-KD) | Full: 10 pp. incl. appendix + 2 pp. refs (2026) | Moderate bar | NV (2026: around May/June) | **Realistic fallback** |
| **AAAI-27 / IJCAI-27** | General AI; TDRO was at AAAI | 7 pp. + 2 pp. refs; reproducibility checklist | Novelty in method; less evaluation depth | AAAI closed; IJCAI paper 11 Jan 2027 | IJCAI: too early. AAAI-28 (≈ Aug 2027) is possible |
| **ACM TORS** | Recommender journal; editorial (2026) reportedly emphasises code and evaluation checklists (secondary source, NV) | Journal | Thoroughness over novelty | Rolling | **Good home for the extended thesis version** |
| **ACM TOIS / IEEE TKDE** | Journals; InvCF follow-up (PopGo) went to TOIS | Journal | Extended experiments | Rolling | Extended version after a conference paper |

**Recommended path.**

1. Thesis.
2. **RecSys 2027 long paper** (≈ April 2027), or **CIKM 2027** if results arrive later.
3. Extended version to **ACM TORS**.

Stretch: KDD 2027 cycle 2 or WWW 2028, if all H1–H8 results are strong.

Reason: the protocol and evaluation contribution (C1) and the reliability analysis are valued at RecSys and TORS. A first-time author's integration-type method is more likely to be judged on rigour there than at KDD or WWW, where reviewers expect a sharper algorithmic novelty.

## 2. Preliminary assessment (pre-experiment)

| Criterion | Rating | Justification |
|---|---|---|
| **Novelty strength** | Moderate | No verified prior work does the coupled *uncertainty ↔ retention ↔ plasticity ↔ cold-start* state in graph rec under a prequential temporal-OOD protocol. However, each primitive is known (risk R1). Novelty rests on the coupling and on the evidence-vs-change test. |
| **Technical contribution** | Moderate–Good | Closed-form uncertainty propagation and GGN for LightGCN; a drift-gated Bayesian-forgetting regulariser; a standardised-surprise statistic with a stated null; an ACI-based selective guarantee. All derivations are short and checkable. |
| **Experimental completeness (planned)** | Good, if executed | Ten experiment families, a coupling ablation, synthetic ground-truth drift, full ranking, multiple seeds, and leakage controls. The risk is scope: it is a lot for one student. |
| **Baseline quality** | Good | It includes the 2024–2025 SOTA in each line (DR-GNN, DRGO, CausalDiffRec, GPT4Rec, GraphPro, DEGC, TDRO, INMO). The risk is GPT4Rec without official code. |
| **Mathematical rigour** | Moderate–Good | The approximations are explicit (`mathematical_formulation.md` §13). The guarantees claimed are only those inherited (ACI) or asymptotic (CLT null). No new theorem is claimed. A reviewer may ask for a bound linking drift gating to forgetting; that is future work. |
| **Reproducibility** | Good (planned) | Public data, released split scripts, fixed seeds, equal tuning budgets. The Amazon-2023 licence wording is NV. |
| **Computational feasibility** | Good | ≈1.3–1.6× LightGCN per block (estimate). Ensembles only on the small dataset. |
| **Main limitations** | — | No interactive or feedback-loop evaluation. Statistical, not causal, invariance. Gaussian/Laplace uncertainty is cruder than ensembles. Calibration depends on the negative-sampling definition. Quarterly blocks miss intra-quarter dynamics. |

**Overall readiness: not yet publishable (there are no results).** The design is credible for RecSys/CIKM-level submission **if** at least H1 (parity or better), H3 (detector) and H4 (frontier) hold, and the coupling ablation shows a clear benefit. If only C1 and C3 hold, the work still supports a solid thesis and possibly a RecSys reproducibility/evaluation-style or workshop paper.

## 3. Refinements recommended before implementation

1. **Sharpen the headline claim to one sentence that a reviewer can falsify.**
   > "A single drift-gated Bayesian node state yields better stability–plasticity trade-offs *and* better-calibrated recommendations under temporal shift than combining separate continual, uncertainty and drift modules."

   Then make the coupling ablation the first experiment run.
2. **Run a 2-week feasibility pilot on Video_Games before building everything.** Use LightGCN + M3 + M5 only. If the surprise detector does not beat raw-loss CUSUM on synthetic drift (H3), revisit the design *before* adding M2, M7 and M8.
3. **Be ready to demote invariance (M1b + V-REx) to an optional module** if the pilot shows no gain over tuned ERM. This is well-documented risk (DomainBed, Wild-Time). The core novelty does not depend on it.
4. **Re-run the literature search** (arXiv cs.IR, SIGIR/KDD/RecSys 2027 accepted lists) one month before submission. Fast-moving adjacent work exists: BAKE (WWW'26, continual KG embedding as sequential Bayesian inference), DRIFT (2026 task-free continual graph learning benchmark), and Doku 2026 on abstention under drift.
5. **Ask the supervisor about the Amazon-2023 licence** and data-sharing policy.
6. **Decide early whether GPT4Rec is re-implemented.** If not, replace it with GPL4SRec (IJCAI'25) or GraphPro as the prompt-based representative, depending on code availability.

## 4. What would make the novelty insufficient (stop conditions)

* A paper appears that maintains Bayesian node posteriors with drift-adaptive forgetting for graph recommendation, with calibration evaluation. Then **pivot the contribution to C1 + C3** (the protocol and the evidence-vs-change detector) and position TIDE-GR as a strong instance.
* The coupling ablation shows no benefit over independent modules. Then the method contribution collapses, and the paper becomes an empirical study ("What matters for reliable continual graph recommendation under temporal shift?"). That is still publishable at RecSys if it is thorough and honest.
