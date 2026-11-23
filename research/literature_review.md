# Stage 2a — Literature Review

## Method and verification status

**Search.** Searches ran on 2026-11-21 using web search and publisher, arXiv, DBLP-via-Semantic-Scholar, OpenReview, NeurIPS/PMLR/AAAI/IJCAI proceedings and GitHub pages. They covered the six categories requested:

* graph OOD;
* recommendation OOD;
* temporal graphs;
* continual recommendation;
* uncertainty;
* open-world.

**Access limitations.**

* DBLP and the ACM DL blocked automated access (HTTP 403). Metadata was therefore confirmed via Semantic Scholar (which returns DBLP keys), arXiv and proceedings pages.
* Google Scholar was not used.
* Some PDFs (IJCAI 2025 proceedings, the GroupLens READMEs) failed to fetch.

**Depth labels.** Each record carries one of these:

* **FT** — the methodology and experiment sections were read (arXiv HTML or PDF text).
* **Abs** — the abstract or proceedings page only.
* **MD** — metadata (title, authors, venue, DOI) verified; the method summary is background knowledge and **not citation-grade** until re-read.

**Status labels.** Peer-reviewed (P), Accepted (A), Preprint (Pre). Reported numbers are the authors' own; none were reproduced.

> The full per-paper evidence for the closest competitors (coverage matrix, threats) is in `closest_competing_methods.md`. This document gives the structured records and the landscape synthesis.

---

## A. Graph OOD

| Paper | Venue / status | Core method | Key finding / limitation | Depth |
|---|---|---|---|---|
| Li, Wang, Zhang, Zhu. *Out-of-Distribution Generalization on Graphs: A Survey* | IEEE TPAMI 47(11), 2025 (P). DOI 10.1109/TPAMI.2025.3593897 | Taxonomy: data, model and learning strategy | Reference survey (note: TPAMI, not TKDE) | Abs |
| Wu, Zhang, Yan, Wipf. **EERM**, *Handling Distribution Shifts on Graphs: An Invariance Perspective* | ICLR 2022 (P). arXiv 2202.02466 | K adversarial graph editors generate virtual environments; REx across them | Single observed environment. Undermined by Chen et al. 2023 | Abs |
| Chen et al. **CIGA** | NeurIPS 2022 (P) | Invariant subgraph via intra-class MI | Graph-level classification | MD |
| Wu et al. **DIR** | ICLR 2022 (P) | Rationale/non-rationale split; interventional environments | Graph-level | MD |
| Gui et al. **GOOD** benchmark | NeurIPS 2022 (P) | Covariate and concept-shift splits | No recommendation tasks | MD |
| Chen, Bian, Zhou, Xie, Han, Cheng. *Does Invariant Graph Learning via Environment Augmentation Learn Invariance?* | NeurIPS 2023 (P). arXiv 2310.19035 | GALA | **Negative result**: augmentation-based invariance is impossible without "variation sufficiency" and "variation consistency" assumptions | Abs |
| Bazhenov et al. *Evaluating Robustness and Uncertainty of Graph Models Under Structural Distributional Shifts* | arXiv 2302.13875 (Pre; NeurIPS'23 NV) | Popularity, locality and density splits | Simple models often beat complex ones under structural shift | Abs |
| Zhang et al. **DIDA** | NeurIPS 2022 (P) | Disentangled dynamic attention; spatio-temporal interventions | Dynamic graphs; not continual | Abs |
| Yuan et al. **EAGLE** | NeurIPS 2023 (P). arXiv 2311.11114 | Environment inference on dynamic graphs; node-wise interventions | Link prediction, not top-K rec | Abs |
| Zhang et al. **SILD** | NeurIPS 2023 (P) | Spectral invariant filtering on dynamic graphs | — | Abs |
| Sun et al. **EvoOOD**, *Evolving Graph Learning for OOD Generalization in Non-stationary Environments* | TPAMI (A, per arXiv comment). arXiv 2511.02354 | Environment sequential VAE + invariant patterns | Closest graph-learning analogue of temporal invariance | Abs |
| Yao et al. **Wild-Time** | NeurIPS 2022 (P). arXiv 2211.14238 | Eval-Fix vs. Eval-Stream temporal benchmark | ~20% ID→OOD drop; **no invariant, continual or SSL method consistently beat ERM** | Abs |
| Sun, Zhang, Ni, Song. **DRIFT** benchmark | arXiv 2605.12998 (Pre, 2026) | Task-free continual graph learning with Gaussian-mixture drift | Continual graph learning methods degrade without task boundaries; node classification only | Abs |

**Foundations of invariance and DRO.**

* Arjovsky et al. **IRM**, arXiv 1907.02893 (Pre, MD).
* Rosenfeld, Ravikumar, Risteski, *The Risks of IRM*, ICLR 2021 (P, MD). IRM needs more environments than spurious dimensions in the linear case, and can fail in the non-linear case.
* Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization*, ICLR 2021 (P, MD). ERM is competitive under fair model selection.
* Krueger et al., *Out-of-Distribution Generalization via Risk Extrapolation (REx)*, ICML 2021 (P, MD). V-REx = Σ R_e + β Var_e R_e.
* Creager et al. **EIIL**, ICML 2021 (P, MD).
* Liu et al. **HRM**, ICML 2021 (P, MD).
* Sagawa et al. **GroupDRO**, arXiv 1911.08731. ICLR 2020 is from memory and was not confirmed by a tool.
* Lu et al. **ICRL**, ICLR 2022 (P, MD).

---

## B. Recommendation OOD, causal and DRO

### B.1 Structured records (most important)

**DR-GNN — *Distributionally Robust Graph-based Recommendation System***

| Field | Value |
|---|---|
| Authors | Bohao Wang, Jiawei Chen, Changdong Li, et al. |
| Year / venue / status | 2024, WWW '24, pp. 3777–3788 (P) |
| DOI / URL | ACM DOI NV; arXiv 2402.12994 |
| Code | github.com/WANGBohaO-jpg/DR-GNN |
| Problem | GNN recommenders degrade under train/test distribution shift |
| Architecture / formulation | LightGCN viewed as a graph-smoothing regulariser. Per-user KL-DRO: $\max_{P:\mathrm{KL}(P,P_u)\le\eta}\mathbb{E}_{v\sim P}[d_u g(u,v)]$ with $g=-E_u^\top E_v/\sqrt{d_ud_v}$. Closed form $P^*(v)\propto P_u(v)e^{g/\alpha}$, which acts as edge reweighting, plus Graph Edge-Addition $\gamma P_u+(1-\gamma)P_{add}$ |
| Datasets | Gowalla, Douban, Amazon-Book, Yelp2018 (popularity); ML-1M, Food (temporal); Coat, Yahoo (exposure) |
| Baselines | LightGCN, InvCF, BOD, LightGCL, SGL, APDA, GAT, GraphDA |
| Metrics | P/R/NDCG@20 |
| Results | Popularity shift: +32–44% NDCG@20 over LightGCN. Temporal: ML-1M 0.2330 vs. LightGCN 0.2041, but APDA 0.2362 is better; Food +3% |
| Strengths | Principled, cheap, closed form |
| Limitations | KL only; sensitive to α; LightGCN only; weak on temporal shift |
| Relevance | Mandatory baseline; worst-case alternative to our edge posterior |
| Depth | FT |

**DRGO — *Distributionally Robust Graph Out-of-Distribution Recommendation via Diffusion Model***

| Field | Value |
|---|---|
| Authors | Chu Zhao, Enneng Yang, Yuliang Liang, et al. |
| Year / venue / status | 2025, WWW '25 (P) |
| DOI / URL | DOI 10.1145/3696410.3714848 (from the uncertainty search pass); arXiv 2501.15555 |
| Code | github.com/user683/DRGO |
| Problem | KL-DRO over-weights noisy samples |
| Method | VGAE latent + latent diffusion denoising; Sinkhorn DRO; entropy regulariser on group weights. Nominal distribution from betweenness-central nodes; K-means uncertainty set; generalisation bound |
| Datasets | Yelp2018, Douban (popularity), Food (temporal), KuaiRec (exposure) |
| Baselines | LightGCN, SGL, SimGCL, LightGCL, InvPref, InvCF, DDRM, AdvInfoNCE, CDR, AdvDrop, DR-GNN |
| Metrics | R/N@10/20 |
| Results | Food R@10 0.0260 → 0.0293. Claims +13.23% over DR-GNN on temporal shift; **some text/table inconsistencies** |
| Limitations | Sensitive to ρ; cost of centrality computation and diffusion |
| Relevance | Mandatory baseline |
| Depth | FT |

**CausalDiffRec — *Graph Representation Learning via Causal Diffusion for Out-of-Distribution Recommendation***

* **Bibliographic:** Zhao et al., WWW 2025 (A/P). arXiv 2408.00490. Code: github.com/user683/CausalDiffRec. **Depth: FT** (appendix not read).
* **DOI:** NV. A Crossref hit "10.145/3696410.3714849" appears malformed and was not used.
* **Method:**
  * SCM with backdoor adjustment;
  * variational environment inference;
  * an RL environment generator producing K perturbed graphs, with a variance loss;
  * a conditional latent DDPM on VGAE latents;
  * LightGCN + BPR.
* **Data:** Food, KuaiRec, Yelp2018, Douban.
* **Results:** the abstract reports average gains of 10.69% (Food) to 22.41% (Yelp). These could not be reconciled with Table 1.
* **Limitations:** it falls below LightGCN on Food and KuaiRec without its generator or inference modules; it overfits with many diffusion steps or environments; it is costly.

**InvCF — *Invariant Collaborative Filtering to Popularity Distribution Shift***

* **Bibliographic:** Zhang, Zheng, Wang, et al. WWW 2023, pp. 1240–1251 (P). DOI 10.1145/3543507.3583461. Code: github.com/anzhang314/InvCF. **Depth: FT.**
* **Method:**
  * separate preference and popularity encoders;
  * popularity-embedding swap via a memory bank, with a consistency loss;
  * distance-correlation disentanglement;
  * loss = softmax + λ1 aug + λ2 disentangle;
  * MF and LightGCN backbones.
* **Data:** Yahoo!R3, Coat, Douban (temporal), Meituan (weekday→weekend), Yelp2018, Tencent.
* **Metrics:** HR/R/N@20.
* **Results:** Douban temporal +2.4–5.9%; Coat R@20 +14.4% (MF).
* **Limitations:** popularity shift only; many hyperparameters.

**AdvDrop** — Zhang, Ma, Wei, et al. WWW 2024 (P). arXiv 2402.13769. Code: github.com/Arthurma71/AdvDrop. **Depth: FT.**

* **Method:** a learnable per-edge bias score drives dropout into bias-aware and bias-mitigated views; InfoNCE invariance; adversarial min-max (ARM estimator).
* **Data:** Yelp2018 (ID/OOD), Yahoo, Coat, KuaiRec, Douban (temporal).
* **Results:** Douban +15.25% NDCG over LightGCN.
* **Limitations:** adversarial instability and cost (stated by the authors).

**PDA — *Causal Intervention for Leveraging Popularity Bias in Recommendation***

* **Bibliographic:** Zhang, Feng, He, et al. SIGIR 2021 (P). arXiv 2105.06067. DOI NV. Code: github.com/zyang1580/PDA. **Depth: FT.**
* **Method:** trains $P(\text{click})=\mathrm{ELU}'(f(u,i))\cdot m_i^\gamma$. PD ranks by the deconfounded $f$. PDA multiplies by forecast popularity $\tilde m=m^T+\alpha(m^T-m^{T-1})$. Drift is measured by JS divergence.
* **Data:** Kwai, Douban, Tencent, with 10 time stages.
* **Results:** PDA beats BPRMF by more than 241% on Kwai.
* **Limitations:** MF only; linear forecast; new users and items excluded.

**TDRO — *Temporally and Distributionally Robust Optimization for Cold-start Recommendation***

* **Bibliographic:** Lin, Wang, Zhao, Li, Feng, Chua. AAAI 2024, 38(8):8750–8758 (P). DOI 10.1609/aaai.v38i8.28721. Code: github.com/Linxyhaha/TDRO. **Depth: FT.**
* **Method:**
  * K-means item groups × E time periods;
  * group selection $j^*=\arg\max[(1-\lambda)\bar L_j+\lambda\langle g_j,\sum_e\beta_e\sum_i g_i^e\rangle]$ with $\beta_e=e^{pe}$;
  * multiplicative weight updates.
* **Data:** Amazon Clothing, Micro-video, Kwai.
* **Baselines:** DUIF, DropoutNet, M2TRec, MTPR, Heater, CB2CF, CCFCRec, CLCRec, GAR, InvRL, S-DRO.
* **Results:** CLCRec backbone, Micro-video R@20 0.0583 → 0.0703 (all), 0.0623 → 0.0761 (cold). Negligible gains with GAR on Kwai cold items.
* **Limitations:** many hyperparameters; hand-crafted groups; depends on features.

### B.2 Further verified recommendation OOD and causal papers

| Paper | Venue / status | Gist | Depth |
|---|---|---|---|
| Zhang et al. **PopGo**, *Robust Collaborative Filtering to Popularity Distribution Shift* | ACM TOIS (P). DOI 10.1145/3627159; arXiv 2310.10696 | Debiasing interaction-wise popularity shortcuts | Abs |
| Wang et al. **InvPref** | KDD 2022 (P). DOI 10.1145/3534678.3539439 | Latent heterogeneous environments; invariant vs. variant preference | Abs |
| Wang et al. **COR**, *Causal Representation Learning for OOD Recommendation* | WWW 2022 (P). DOI 10.1145/3485447.3512251 | Causal VAE for user-feature shift; post-intervention inference | Abs |
| He et al. **CausPref** | WWW 2022 (P). arXiv 2202.03984 | Causal preference learning for OOD | MD |
| Yang et al. **DROS**, *A Generic Learning Framework for Sequential Recommendation with Distribution Shifts* | SIGIR 2023 (P). DOI 10.1145/3539618.3591624 | DRO for sequential recommendation, aimed at future data | Abs |
| Wen et al. **S-DRO**, *Distributionally-robust Recommendations for Improving Worst-case User Experience* | WWW 2022 short (P) | Group DRO with streaming loss estimates | Abs |
| Liang et al. **InvGCLLM** | arXiv 2511.18282 (Pre) | LLM-judged causal edges + invariant contrastive learning on LightGCN | FT |
| Zhu et al. **CICDOR** | arXiv 2505.16532 (Pre; arXiv comment says accepted to TOIS, NV) | Causal-invariant cross-domain OOD rec | Abs |
| Wang et al. **CILER** | arXiv 2608.03647 (Pre, 2026) | Identifiable latent-environment OOD rec; risk-aware marginalisation | Abs |
| Zhu et al. **ELBO_TDS** | arXiv 2511.21032 (Pre) | Probabilistic temporal distribution generalisation, Shopee; +2.33% GMV/user (abstract) | Abs |
| Zheng et al. **DICE** | WWW 2021 (P). DOI 10.1145/3442381.3449788 | Interest vs. conformity disentanglement | MD |
| Wei et al. **MACR** | KDD 2021 (P). DOI 10.1145/3447548.3467289 | Counterfactual removal of popularity effect | MD |
| Wang et al. **DGCF** | SIGIR 2020 (P). DOI 10.1145/3397271.3401137 | Intent-disentangled graph CF | MD |

**Surveys:**

* Zhang et al., *Robust Recommender System: A Survey and Future Directions*, arXiv 2309.02057. Covers attacks and noise, not OOD.
* Gao et al., *Causal Inference in Recommender Systems: A Survey and Future Directions*, TOIS (per arXiv 2208.12397).
* Luo et al., *A Survey on Causal Inference for Recommendation*, arXiv 2303.11666.

**No dedicated OOD-recommendation survey was found.**

---

## C. Temporal and dynamic graph learning

| Paper | Venue | Gist | Relevance | Depth |
|---|---|---|---|---|
| Rossi et al. **TGN** | arXiv 2006.10637 | Event memory + graph operators | Event-based alternative (S2) | Abs |
| Xu et al. **TGAT** | ICLR 2020 | Functional time encoding + attention | Event-based alternative | MD |
| Kumar, Zhang, Leskovec. **JODIE** | KDD 2019. DOI 10.1145/3292500.3330895 | Coupled RNN embedding trajectories | Temporal rec ancestor | MD |
| Fan et al. **TGSRec** | CIKM 2021. arXiv 2108.06625 | Temporal collaborative transformer | Repository README reports bugs and reruns | Abs |
| Poursafaei et al. *Towards Better Evaluation for Dynamic Link Prediction* (**EdgeBank**) | NeurIPS 2022 D&B | Memorisation baseline; harder negatives | Lenient negatives inflate temporal-GNN results | Abs |
| Cong et al. **GraphMixer** | ICLR 2023 | MLP link encoder + mean pooling | Complex temporal architectures not necessary | Abs |
| Huang et al. **TGB** | NeurIPS 2023 D&B | Temporal Graph Benchmark | Simple methods often superior (node tasks) | Abs |
| Yu et al. **DyGFormer/DyGLib** | NeurIPS 2023 | Unified library; some baselines irreproducible | — | Abs |
| Gultekin & Paisley. *A Collaborative Kalman Filter for Time-Evolving Dyadic Processes* | arXiv 1501.05624 (ICDM 2014 per one listing, NV) | Brownian-motion MF embeddings; mean-field VI; ML-10M, Netflix | **Direct ancestor of our state-space view** (MF, not graph; no calibration or cold-start) | Abs |

---

## D. Continual and streaming recommendation

**GPT4Rec — *Graph Prompt Tuning for Streaming Recommendation***

* **Bibliographic:** Zhang, Yan, Zhang, et al. SIGIR 2024 (P). DOI 10.1145/3626772.3657720. Code not linked. **Depth: Abs + experiments.**
* **Method:** disentangled graph views with node-, structure- and view-level prompts on a frozen backbone.
* **Data:** Netflix, Foursquare, Taobao2014/15.
* **Baselines:** Finetune, Uniform/Inverse-Degree replay, ContinualGNN, TWP, GraphSAIL, SGCT, MGCT, LWC-KD, DEGC.
* **Metrics:** R/N@20.
* **Results:** Taobao2014 R@20 0.1127 vs. DEGC 0.1082.
* **Limitations:** small gains; no shift modelling; text inconsistency (six vs. four datasets).

**GraphPro — *Graph Pre-training and Prompt Learning for Recommendation***

* **Bibliographic:** Yang, Xia, Luo, et al. WWW 2024 (P). arXiv 2311.16716. Code: github.com/HKUDS/GraphPro. **Depth: Abs + experiments.**
* **Method:** temporal prompt + graph-structural prompt; snapshot adaptation.
* **Data:** Taobao (daily), Koubei and Amazon (weekly).
* **Results:** Taobao/LightGCN R@20 0.0251 vs. EvolveGCN-O 0.0236; online CTR +1.53%.
* **Limitations:** sensitive to the update window; no forgetting metrics.

**DEGC — *Dynamically Expandable Graph Convolution for Streaming Recommendation***

* **Bibliographic:** He, He, Zhang, et al. WWW 2023 (P). DOI 10.1145/3543507.3583237. Code: github.com/BokwaiHo/DEGC. **Depth: Abs + experiments.**
* **Method:** prune, refine and expand graph convolutions; temporal user-embedding initialisation.
* **Data:** Taobao2014/15, Netflix, Foursquare.
* **Backbones:** NGCF, LightGCN, MGCCF.
* **Metrics:** R/N@20.

**Other continual and streaming work:**

| Paper | Venue | Gist | Depth |
|---|---|---|---|
| Xu et al. **GraphSAIL** | CIKM 2020. DOI 10.1145/3340531.3412754 | Local/global structure distillation | Abs |
| Ahrabian et al. **SAER**, *Structure Aware Experience Replay* | CIKM 2021. DOI 10.1145/3459637.3482193 | Structure-aware reservoir replay | Abs |
| Wang, Zhang, Coates. **LWC-KD** | CIKM 2021. DOI 10.1145/3459637.3482117 | Contrastive KD for incremental graph rec | MD |
| Wang et al. **PIW**, *Structure Aware Incremental Learning with Personalized Imitation Weights* | AAAI 2023 | Per-user imitation weight (shifting users imitate less) | Abs |
| Cang et al. **GPL4SRec** | IJCAI 2025. DOI 10.24963/ijcai.2025/302 | Multi-level graph prompts for streaming rec | MD |
| Wang et al. **ContinualGNN** | CIKM 2020. DOI 10.1145/3340531.3411963 | Replay + weighted regularisation for streaming GNNs | Abs |
| Mi et al. **ADER** | RecSys 2020 | Exemplar replay + adaptive distillation (session rec) | Abs |
| Do & Lauw. *Continual Collaborative Filtering Through Gradient Alignment* | RecSys 2023 | Gradient alignment across user/item tasks | Abs |
| Lee et al. **CCD**, *Continual Collaborative Distillation* | KDD 2024 | Teacher–student co-evolution over the stream | Abs |
| Valkanas et al. *Personalized Negative Reservoir* | arXiv 2403.03993 (venue NV) | Per-user negative reservoir | Abs |
| Fan et al. *Light distillation for incremental graph CF* | arXiv 2505.19810 (Pre) | Score distillation | Abs |
| Yoo, Kang, Tong. *Continual Recommender Systems* (tutorial) | arXiv 2507.03861 (CIKM 2025 tutorial per snippet, NV) | Per-user stability/plasticity; retraining cost | Abs |
| Bisht et al. **ALIGN-Rec** | NeurIPS 2026 (A) | Continual rec with unlearning | Abs |
| Zhang, Song, Tao. **CGLB** | NeurIPS 2022 D&B | Continual graph learning benchmark (no rec) | Abs |

**Continual-learning foundations:**

* Kirkpatrick et al. **EWC**, PNAS 2017, DOI 10.1073/pnas.1611835114 (MD).
* Li & Hoiem **LwF**, ECCV 2016 / TPAMI (MD).
* Chaudhry et al., *On Tiny Episodic Memories*, arXiv 1902.10486 (Abs). Joint training with tiny memory beats specialised continual-learning methods.
* Lopez-Paz & Ranzato **GEM**, NeurIPS 2017 (Abs). BWT/FWT metrics.
* **Bayesian continual learning** (verified in this session):
  * Nguyen et al. **VCL**, ICLR 2018 (Abs);
  * Ritter, Botev, Barber, *Online Structured Laplace Approximations*, NeurIPS 2018 (Abs). Kronecker-factored online Laplace; reviewers note EWC is a special case;
  * **Kurle et al., *Continual Learning with Bayesian Neural Networks for Non-Stationary Data*, ICLR 2020 (Abs).** Diagonal Gaussian + raw-data memory for poorly approximated likelihood terms, plus **Bayesian forgetting / Gaussian diffusion** for non-stationarity.
* BAKE (WWW 2026, DOI 10.1145/3774904.3792692): continual KG embedding as sequential Bayesian inference, with the posterior used as the next prior (Abs, search snippet only).

---

## E. Uncertainty

**Recommendation:**

| Paper | Venue | Gist | Depth |
|---|---|---|---|
| Kweon, Kang, Yu. *Obtaining Calibrated Probabilities with Personalized Ranking Models* | AAAI 2022. DOI 10.1609/aaai.v36i4.20326 | Gaussian/Gamma post-hoc calibration with IPS-unbiased fitting; ECE/MCE/NLL | Abs |
| Kweon et al. **PerK**, *Top-Personalized-K Recommendation* | WWW 2024. DOI 10.1145/3589334.3645417 | Calibrated probabilities → per-user K | Abs |
| Coscrato & Bridge. *Estimating and Evaluating the Uncertainty of Rating Predictions and Top-n Recommendations* | ACM TORS 1(2), 2023 | Survey + new estimators and metrics | Abs |
| Angelopoulos et al. *Recommendation Systems with Distribution-Free Reliability Guarantees* | COPA 2023, PMLR 204 | Learn-then-Test FDR control; assumes exchangeability | Abs |
| Sun et al. **BGCN-Rec**, *A Framework for Recommending Accurate and Diverse Items Using Bayesian Graph Convolutional Neural Networks* | **KDD 2020** (not CIKM). DOI 10.1145/3394486.3403254 | Random-graph posterior, node copying | Abs |
| Tian et al. **RGCF** | SIGIR 2022. DOI 10.1145/3477495.3531889 | Edge denoising + reweighting | Abs |
| Wang et al. **ADT (T-CE / R-CE)**, *Denoising Implicit Feedback* | **WSDM 2021** (not MM). DOI 10.1145/3437963.3441800 | Truncated / reweighted loss on high-loss interactions | Abs |
| Liu et al. **UCC** | SIGIR 2023 short. DOI 10.1145/3539618.3592078 | Uncertainty-aware consistency for cold items | Abs |
| Li et al. **W-GAT** | IEEE TCSS 2024 (per arXiv 2404.05962) | Gaussian embeddings + Wasserstein graph attention | Abs |
| Neupane, Zheng, Yu. **E-NSDE** | NeurIPS 2024 | Evidential neural SDE, time-aware sequential rec | Abs |
| Kweon et al. *Uncertainty Quantification and Decomposition for LLM-based Recommendation* | WWW 2025 | Uncertainty decomposition for LLM rec | Abs |
| Wang et al. *Confidence-aware Fine-tuning of Sequential Rec via Conformal Prediction* | arXiv 2402.08976 (venue NV) | Conformal losses in sequential rec | Abs |
| Li et al. **LiDu**, *Are Recommenders Self-Aware?* | arXiv 2507.23208 (Pre) | List-level uncertainty as a label-free performance proxy | Abs |
| Zou et al. *Uncertainty-Calibrated Recommendations for Low-Active Users* | arXiv 2605.17788 (KDD 2026 ADS per arXiv, NV) | Error critic; risk-averse vs. UCB by user activity | Abs |
| Doku. *The Confidence Gate Theorem* | arXiv 2603.09947 (Pre) | Abstention helps under structural (cold-start) uncertainty, can fail under contextual drift | Abs |

**Graph and general uncertainty foundations:**

| Paper | Venue | Gist | Depth |
|---|---|---|---|
| Zhang et al. *Bayesian GCN for Semi-Supervised Classification* | AAAI 2019. DOI 10.1609/aaai.v33i01.33015829 | Graph as a random-graph sample (MMSBM) | Abs |
| Zhao et al. **GKDE** | NeurIPS 2020 | Vacuity / dissonance / aleatoric / epistemic | Abs |
| Stadler et al. **GPN**, Graph Posterior Network | NeurIPS 2021 | Flow pseudo-counts + APPNP evidence propagation | Abs |
| Huang et al. **CF-GNN** | NeurIPS 2023 | Conformalised GNNs; up to 74% smaller sets | Abs |
| Clarkson. *Distribution Free Prediction Sets for Node Classification* | ICML 2023 | Neighbourhood-weighted conformal | Abs |
| Barber, Candès, Ramdas, Tibshirani. *Conformal Prediction Beyond Exchangeability* | Annals of Statistics 51(2), 2023. DOI 10.1214/23-AOS2276 | Weighted quantiles; coverage gap ≤ Σ w_i d_TV | Abs |
| **Gibbs & Candès. *Adaptive Conformal Inference Under Distribution Shift*** | NeurIPS 2021 | $\alpha_{t+1}=\alpha_t+\gamma(\alpha-\mathrm{err}_t)$; long-run coverage for any sequence | Abs |
| Angelopoulos, Candès, Tibshirani. *Conformal PID Control* | NeurIPS 2023 | Generalises ACI | Abs |
| **Ovadia et al. *Can You Trust Your Model's Uncertainty?*** | NeurIPS 2019 | Ensembles most robust under shift; temperature scaling can hurt under shift | Abs |
| Daxberger et al. **Laplace Redux** | NeurIPS 2021 | Post-hoc Laplace competitive at low cost | Abs |
| Bengs, Hüllermeier, Waegeman. *Pitfalls of Epistemic UQ through Loss Minimisation* | NeurIPS 2022 | Second-order (evidential) predictors are not faithful | Abs |
| Meinert, Gawlikowski, Lavin. *The Unreasonable Effectiveness of Deep Evidential Regression* | AAAI 2023. DOI 10.1609/aaai.v37i8.26096 | DER is heuristic | Abs |

Not re-verified in this session (citations from memory; the DOIs must be checked before citing):

* Lakshminarayanan et al., NeurIPS 2017 (deep ensembles)
* Gal & Ghahramani, ICML 2016 (MC dropout)
* Kendall & Gal, NeurIPS 2017
* Depeweg et al., ICML 2018
* Guo et al., ICML 2017
* Sensoy et al., NeurIPS 2018
* Kipf & Welling 2016 (VGAE)
* MacKay 1992 (probit approximation)
* Maddison et al. / Jang et al. 2017 (Concrete / Gumbel-softmax)
* Page 1954 (CUSUM)
* Efraimidis & Spirakis 2006 (weighted reservoir)

---

## F. Open-world, inductive and cold-start

**IDCF — *Towards Open-World Recommendation: An Inductive Model-based Collaborative Filtering Approach***

* **Bibliographic:** Wu, Zhang, Gao, Yan, Zha. ICML 2021 (P). arXiv 2007.04833. Code: github.com/qitianwu/IDCF. **Depth: FT (partial).**
* **Method:** MF on key users gives meta latents. An attention-estimated relation graph between query and key users drives message passing, so query users get embeddings without retraining.
* **Data:** Douban, ML-100K, ML-1M, Amazon-Books, Amazon-Beauty.
* **Metrics:** RMSE/NDCG; AUC/NDCG with sampled negatives.
* **Limitations:** attention over all key users does not scale (stated); sampled-negative evaluation.

**INMO — *A Model-Agnostic and Scalable Module for Inductive Collaborative Filtering***

* **Bibliographic:** Wu, Cao, Shen, Tao, Cheng. SIGIR 2022 (P). DOI 10.1145/3477495.3532000. Code: github.com/WuYunfan/igcn_cf. **Depth: FT.**
* **Method:**
  * $e_u=(|N_u\cap I_{tem}|+1)^{-\alpha}(\sum_{i\in N_u\cap I_{tem}}t_i+t_{user})$;
  * error-sort template selection;
  * $L=L_{BPR}+\beta L_{SE}$.
* **Data:** Gowalla, Yelp, Amazon-Book.
* **Metrics:** full-ranking R/P/NDCG@20.
* **Results:** new users/items NDCG@20 ×100 on Gowalla: INMO-LGCN 14.52 vs. IDCF-LGCN 9.40, with a full-retrain bound of 15.41.
* **Limitations:** templates fixed at training time; random (non-temporal) holdout.

**Other cold-start and inductive work:**

| Paper | Venue | Depth |
|---|---|---|
| Zhang & Chen **IGMC** | ICLR 2020 | MD |
| Ying et al. **PinSage** | KDD 2018. DOI 10.1145/3219819.3219890 | MD |
| Hamilton et al. **GraphSAGE** | NeurIPS 2017 | MD |
| Volkovs et al. **DropoutNet** | NeurIPS 2017 | MD |
| Lee et al. **MeLU** | KDD 2019. DOI 10.1145/3292500.3330859 | MD |
| Zhu et al. **Heater** | SIGIR 2020. DOI 10.1145/3397271.3401178 | MD |
| Wei et al. **CLCRec** | MM 2021. DOI 10.1145/3474085.3475665 | MD |
| Huang et al. **ALDI** | SIGIR 2023. DOI 10.1145/3539618.3591732 | MD |
| Bai et al. **GoRec** | MM 2023. DOI 10.1145/3581783.3612238 | MD |
| Lu et al. **MetaHIN** | KDD 2020. DOI 10.1145/3394486.3403207 | MD |
| Huang et al. **ColdLLM** | WSDM 2025. DOI 10.1145/3701551.3703546 | Abs |
| Liu et al. **USIM** | NeurIPS 2024. DOI 10.52202/079017-0284 | Abs |

---

## G. Graph CF encoders and evaluation methodology

**Encoders** (all MD unless noted):

* **LightGCN**, SIGIR 2020, DOI 10.1145/3397271.3401063
* **NGCF**, SIGIR 2019, DOI 10.1145/3331184.3331267
* **SGL**, SIGIR 2021, DOI 10.1145/3404835.3462862
* **SimGCL**, SIGIR 2022, DOI 10.1145/3477495.3531937
* **XSimGCL**, TKDE 2023, DOI 10.1109/TKDE.2023.3288135
* **LightGCL**, ICLR 2023
* **GFormer**, SIGIR 2023, DOI 10.1145/3539618.3591723
* **SIGformer**, SIGIR 2024, DOI 10.1145/3626772.3657747
* Cai et al., *How Expressive are GNNs in Recommendation?*, CIKM 2023, DOI 10.1145/3583780.3614917

**Methodology:**

* Ferrari Dacrema et al., RecSys 2019, DOI 10.1145/3298689.3347058
* Shehzad & Jannach, *Everyone's a Winner!*, RecSys 2023, DOI 10.1145/3604915.3609488
* Milogradskii et al., *Revisiting BPR*, RecSys 2024, DOI 10.1145/3640457.3688073
* Ji et al., *A Critical Study on Data Leakage in RecSys Offline Evaluation*, TOIS 2023, DOI 10.1145/3569930
* Meng et al., *Exploring Data Splitting Strategies*, RecSys 2020, DOI 10.1145/3383313.3418479
* Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020, DOI 10.1145/3394486.3403226
* Woolridge et al., *Sequence or Pseudo-Sequence?*, Perspectives@RecSys 2021, CEUR-WS Vol-2955 (FT; MovieLens timestamp artefacts)

---

## H. Synthesis

1. **Three separate lines exist and have not merged** (see `closest_competing_methods.md` §3):
   * static OOD graph rec (DR-GNN → DRGO / CausalDiffRec);
   * streaming graph rec (GraphSAIL → PIW → DEGC → GPT4Rec / GraphPro / GPL4SRec);
   * uncertainty in rec (calibration, conformal, BGCN-Rec).

   Dynamic-graph OOD (DIDA / EAGLE / EvoOOD) is a fourth line outside recommendation.
2. **Temporal shift in OOD rec is usually a single static split** (Food, ML-1M, Douban). Only PDA and TDRO model temporal trends, and neither is a graph method.
3. **Uncertainty under drift is the least-developed axis** for graph rec. No calibrated-uncertainty graph recommender evaluated under temporal shift was found. Generic online-calibration tools (ACI, Conformal PID) and Bayesian continual learning (VCL, online Laplace, Kurle et al.) are mature but unused there.
4. **Strong negative evidence must shape the design:**
   * invariance may not beat ERM (DomainBed, Wild-Time, Chen et al. 2023);
   * complex temporal encoders may not beat simple ones (EdgeBank, GraphMixer, TGB);
   * evidential epistemic estimates are contested (Bengs et al., Meinert et al.);
   * MovieLens timestamps are unreliable (Woolridge et al.);
   * global k-core filtering and random splits leak (Ji et al.).
5. These observations led to the simple backbone, the coupling-centred contribution, and the evaluation design.
