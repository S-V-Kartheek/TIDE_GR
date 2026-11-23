# Stage 7c — Baselines

* **"Repo"** gives the code location reported by the paper or a verified GitHub page.
* **NV** = the repository was not verified to be official or runnable.
* **"Original data/protocol"** is what the paper reports.

All baselines are **re-run under our prequential protocol** (`experimental_design.md` §0). Static methods run in both retrain-per-block and fine-tune-per-block modes.

## Tier 1 — Closest competitors (mandatory)

| Baseline | Paper / venue | Repo | Original data / protocol | Why included |
|---|---|---|---|---|
| **DR-GNN** | Wang et al., *Distributionally Robust Graph-based Recommendation System*, WWW 2024 | github.com/WANGBohaO-jpg/DR-GNN | Gowalla, Douban, Amazon-Book, Yelp2018 (popularity); ML-1M, Food (temporal); Coat, Yahoo (exposure); P/R/NDCG@20 | Strongest graph DRO OOD recommender; tested on temporal shift |
| **DRGO** | Zhao et al., *Distributionally Robust Graph Out-of-Distribution Recommendation via Diffusion Model*, WWW 2025, DOI 10.1145/3696410.3714848 | github.com/user683/DRGO | Yelp2018, Douban, Food, KuaiRec; R/N@10/20 | Latest graph OOD rec (Sinkhorn DRO + diffusion) |
| **CausalDiffRec** | Zhao et al., *Graph Representation Learning via Causal Diffusion for OOD Recommendation*, WWW 2025 | github.com/user683/CausalDiffRec | Food, KuaiRec, Yelp2018, Douban | Invariant / causal graph OOD rec with environment inference |
| **InvCF** | Zhang et al., *Invariant Collaborative Filtering to Popularity Distribution Shift*, WWW 2023, DOI 10.1145/3543507.3583461 | github.com/anzhang314/InvCF | Yahoo!R3, Coat, Douban (temporal), Meituan, Yelp2018, Tencent; HR/R/N@20 | Preference/popularity invariance on a LightGCN backbone |
| **GPT4Rec** | Zhang et al., *Graph Prompt Tuning for Streaming Recommendation*, SIGIR 2024, DOI 10.1145/3626772.3657720 | none linked (NV) | Netflix, Foursquare, Taobao2014/15; R/N@20 over segments | SOTA streaming graph rec. If no code is available, re-implement from the paper and state this |
| **GraphPro** | Yang et al., *Graph Pre-training and Prompt Learning for Recommendation*, WWW 2024 | github.com/HKUDS/GraphPro | Taobao, Koubei, Amazon snapshots; R/N@20 | Temporal-prompt dynamic graph rec |
| **DEGC** | He et al., *Dynamically Expandable Graph Convolution for Streaming Recommendation*, WWW 2023, DOI 10.1145/3543507.3583237 | github.com/BokwaiHo/DEGC | Taobao2014/15, Netflix, Foursquare; R/N@20 | Model-isolation streaming graph rec |
| **TDRO** | Lin et al., *Temporally and Distributionally Robust Optimization for Cold-start Recommendation*, AAAI 2024, DOI 10.1609/aaai.v38i8.28721 | github.com/Linxyhaha/TDRO | Amazon Clothing, Micro-video, Kwai; R/N@20 over all / warm / cold | Temporal + cold-start; open-world track |

## Tier 2 — Strong component baselines

| Baseline | Paper / venue | Repo | Why |
|---|---|---|---|
| **LightGCN** (retrain-per-block / fine-tune) | He et al., SIGIR 2020, DOI 10.1145/3397271.3401063 | github.com/gusye1234/LightGCN-PyTorch (NV as official; the official TF code is from the authors' group) | IID backbone; the "no-mechanism" reference |
| **BPR-MF** | Rendle et al., UAI 2009 | standard implementations | Sanity baseline; replicability (Milogradskii et al., RecSys'24) |
| **SimGCL** | Yu et al., SIGIR 2022, DOI 10.1145/3477495.3531937 | github.com/Coder-Yu/SELFRec (NV) | Strong static graph CF; checks that gains are not just from capacity |
| **PDA** | Zhang et al., SIGIR 2021 | github.com/zyang1580/PDA | Popularity-drift forecasting (MF) |
| **GraphSAIL** | Xu et al., CIKM 2020, DOI 10.1145/3340531.3412754 | NV | Distillation-based incremental graph rec |
| **LWC-KD** / **SAER** | Wang et al. / Ahrabian et al., CIKM 2021 | NV | Distillation / replay incremental graph rec |
| **PIW** | Wang et al., AAAI 2023 | NV | Per-user stability/plasticity: closest to our per-node gating |
| **INMO** | Wu et al., SIGIR 2022, DOI 10.1145/3477495.3532000 | github.com/WuYunfan/igcn_cf | Inductive new users/items |
| **IDCF** | Wu et al., ICML 2021 | github.com/qitianwu/IDCF | Inductive open-world CF (if it scales) |
| **ALDI** | Huang et al., SIGIR 2023, DOI 10.1145/3539618.3591732 | NV | Strong cold-item content baseline |
| **RGCF** | Tian et al., SIGIR 2022, DOI 10.1145/3477495.3531889 | NV | Edge denoising (structural uncertainty comparator) |
| **T-CE / R-CE (ADT)** | Wang et al., WSDM 2021, DOI 10.1145/3437963.3441800 | github.com/WenjieWWJ/DenoisingRec | Loss-based interaction reliability |

## Tier 3 — Uncertainty comparators (same backbone)

| Comparator | Reference | Purpose |
|---|---|---|
| Deep ensemble (M=5) | Lakshminarayanan et al., NeurIPS 2017 (citation from memory; re-verify) | Upper reference for epistemic quality under shift (Ovadia et al., NeurIPS 2019) |
| MC dropout | Gal & Ghahramani, ICML 2016 | Cheap baseline |
| Temperature scaling | Guo et al., ICML 2017 | Static calibration |
| Gaussian/Gamma calibration | Kweon et al., AAAI 2022, DOI 10.1609/aaai.v36i4.20326 | Static rec calibration |
| Split conformal / Learn-then-Test | Angelopoulos et al., COPA 2023 (PMLR 204) | Exchangeable conformal rec (contrast with ACI) |
| Raw-loss ADWIN/CUSUM, embedding displacement, MMD | — | Drift-detector comparators |

## Fairness rules

1. Same data, splits, negatives and candidate sets for all methods.
2. Same embedding dimension (64) where applicable, and the same epoch or early-stopping policy.
3. The same tuning budget (trials) per method, on the warm-up period only.
4. Static OOD methods get both update modes, and we report the better of the two.
5. When a baseline cannot run at Toys_and_Games scale (e.g., CausalDiffRec diffusion, IDCF attention), report it on Video_Games and state the reason.
6. Report our re-implementation status per baseline: official code / adapted / re-implemented.

**Excluded (with reason).**

* MovieLens-specific temporal baselines.
* LLM-based cold-start (ColdLLM WSDM'25, USIM NeurIPS'24): different resource class and a contamination risk. They are cited, not run.
* TGN, TGSRec and DyGFormer: not run as main baselines because of cost and documented reproducibility issues (the TGSRec repository notes bugs). An optional single run on Video_Games is planned if time permits.
