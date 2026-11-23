# Stage 7a — Dataset Analysis

All statistics below are copied from official dataset pages or dataset papers, checked on 2026-11-21. **NV** = not verified from a primary source; do not cite an NV figure until it has been checked.

## 1. Requirements derived from the research question

| Requirement | Why |
|---|---|
| Per-interaction timestamps with meaningful order | Temporal blocks, prequential evaluation, drift |
| Long span (≥ several years), or many blocks | Need ≥ 8–12 blocks to measure OOD *sequences*, forgetting and seasonality |
| Item metadata (text/category) | Open-world items need a feature prior (Challenge E) |
| Steady arrival of new users/items | Open-world evaluation |
| Observable spurious factors (seasonality, popularity bursts) | Challenge B must be testable |
| Public, research-licensed | Reproducibility |
| Moderate size (10^5–10^7 interactions per experiment) | Final-year compute |

## 2. Comparison

| Dataset | Users | Items | Interactions | Timestamps | Metadata | Temporal suitability | Open-world suitability |
|---|---|---|---|---|---|---|---|
| **Amazon Reviews 2023** (all) | 54.51M | 48.19M | 571.54M | ms; May 1996 – Sep 2023 | Item: title, features, description, price, images, store, categories, details. User: none | **High.** 27-year span; seasonality (gift categories); official absolute-timestamp split | **High.** Continuous item and user arrival; rich item text |
| ↳ Video_Games (full / 5-core) | 2.8M / 94.8K | 137.2K / 25.6K | 4.6M / 814.6K | as above | as above | High | High |
| ↳ Toys_and_Games (full / 5-core) | 8.1M / 432.3K | 890.7K / 162.0K | 16.3M / 3.9M | as above | as above | **High; strong holiday seasonality** (hypothesis to be checked in EDA) | High |
| ↳ Beauty_and_Personal_Care (full / 5-core) | 11.3M / 729.6K | 1.0M / 207.6K | 23.9M / 6.6M | as above | as above | High | High |
| **Yelp Open Dataset** | NV | 150,346 businesses | 6,990,280 reviews | Review dates (granularity and span NV) | Business attributes, categories, location. User fields NV | Medium–High (multi-year; span NV) | Medium–High (new businesses; geographic metadata) |
| **MovieLens-1M** | 6,040 | ~3,900 | 1,000,209 | s, from 2000 | Title, genres; user demographics (NV this session) | **Low.** 59% of users rated everything on one date (Woolridge et al., 2021) | Low |
| **MovieLens-32M** | 200,948 | 87,585 | 32,000,204 | s; Jan 1995 – Oct 2023 | Title, genres, tags | Medium. Long span but bulk-entry artefact (56.4% single-date users in ML-25M) | Medium |
| **MIND** | 1,000,000 | 161,013 | 24,155,470 clicks / 15.78M impressions | s; 12 Oct – 22 Nov 2019 (6 weeks) | News title, abstract, category, entities | Low for long-term drift; good for very short-term drift | High (news items are almost all new) but it is cold-start-only by construction |
| **Last.fm-1K** | 992 | 107,528 artists with MBID (+69,420 without) | 19,150,868 events | per event, to May 2009 | Artist/track names. User: gender, age, country | Medium (long histories, few users) | Low (992 users) |
| LFM-1b / LFM-2b | — | — | — | — | — | **Withdrawn** ("license issues") | — |
| **KuaiRand-27K** | 27,285 | 32,038,725 | 322,278,385 | ms; 8 Apr – 8 May 2022 | Video and user features | Low (1 month) | Medium; useful because of randomly exposed slice (debiased evaluation) |
| **Goodreads (UCSD Book Graph)** | 876,145 | 2,360,655 | 229,154,523 | date_added / read_at etc.; collected 2017 | Book, author, series | Medium–High | Medium |
| Netflix Prize | >480K (paper) | ~18K | >100M | **date only**; 1998–2005 | Title, year | Medium | Low. **No official distribution now** |
| Taobao UserBehavior | NV | NV | NV | NV (≈9 days, secondary sources) | Category | Low (days) | Low |

Sources:

* Amazon Reviews 2023: amazon-reviews-2023.github.io (homepage and 5-core page). Dataset paper: Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952. The arXiv comment says ACL 2026.
* Yelp: business.yelp.com/data/resources/open-dataset/
* MovieLens: grouplens.org; Harper & Konstan, ACM TiiS 2015.
* MovieLens timestamp caveat: Woolridge et al., Perspectives@RecSys 2021, CEUR-WS Vol-2955.
* MIND: Wu et al., ACL 2020, DOI 10.18653/v1/2020.acl-main.331.
* Last.fm-1K: Zenodo record 6090214.
* LFM-1b / LFM-2b: cp.jku.at.
* KuaiRand: kuairand.com; Gao et al., CIKM 2022.
* Goodreads: UCSD Book Graph page.
* Netflix Prize: Bennett & Lanning, KDD Cup 2007.

## 3. Licensing and access

| Dataset | Terms |
|---|---|
| Amazon Reviews 2023 | Licence not stated on the homepage (**NV**). Treat as research use and cite the dataset paper; check with the supervisor before any redistribution of derived splits. Release split *scripts*, not data. |
| Yelp | Academic, non-commercial, revocable, non-sublicensable. You must agree to the Dataset User Agreement (17 Jul 2023 version). No redistribution. |
| MovieLens | Research use; no public redistribution without permission; commercial use needs permission. |
| MIND | Microsoft Research License Terms; you must agree before download. |
| KuaiRand / KuaiRec | CC BY-SA 4.0. |
| Last.fm-1K | Non-commercial (Zenodo "Other (Non-Commercial)"). |
| Goodreads | Academic use only; no redistribution. |

## 4. Recommendation

### A. Primary dataset — **Amazon Reviews 2023, Toys_and_Games** (built from the *full* file, not the official 5-core)

* The only public dataset with all of: ms timestamps over more than two decades, continuous arrival of new users and items, and rich item text for cold-start priors.
* Toys are expected to show **strong seasonal/holiday shifts**. That provides naturally occurring *spurious temporal correlations* (holiday gift purchases versus a stable interest in, e.g., board games), which is exactly Challenge B. *This expectation must be confirmed in EDA by plotting monthly volume and category mass. Until then it is a hypothesis.*
* It is used by the BLaIR / Amazon-2023 benchmark community, which helps comparability.
* Working window: **2015-01 → 2023-09**, about 105 monthly blocks; we aggregate into **quarterly blocks**, giving about 35 blocks. Restricting the window keeps the size tractable and avoids the sparse early years. Exact counts after filtering must be computed and reported, not assumed.

### B. Secondary, cross-domain dataset — **Yelp Open Dataset**

* Different domain (local services), different drift drivers (openings and closures, local trends), business metadata for cold items, and a multi-year span.
* The license is academic-only, which is acceptable for a thesis. The span and user count must be verified on download (**NV**).
* *Alternative if Yelp is unsuitable after EDA:* Goodreads (academic use; reading dates; book metadata).

### C. Small initial experimentation dataset — **Amazon Reviews 2023, Video_Games**, temporally filtered

* Full file: 4.6M ratings. After per-block filtering it is expected to be well under 1M, so prototyping is fast.
* Same schema as the primary dataset, so the pipeline transfers.
* Used for the **deep-ensemble upper-reference** and for **controlled synthetic-drift** experiments, because it is small enough for repeated runs.

### Explicitly *not* used for temporal claims

* **MovieLens (any version).** Timestamps record rating-entry time, with bulk entry at signup (Woolridge et al., 2021). MovieLens-1M may be used **only** as a sanity check for the static ranking pipeline, and never for OOD or forgetting claims.
* **MIND and KuaiRand.** Their spans are too short for long-horizon drift. KuaiRand's randomly exposed slice could support an optional exposure-bias study, but that is out of scope.

### Comparability with streaming-graph-rec baselines

DEGC and GPT4Rec use Taobao2014/2015, Netflix and Foursquare. GraphPro uses Taobao, Koubei and Amazon.

* The official Netflix Prize distribution is no longer available.
* The Taobao variants' exact provenance is **NV**.

We therefore run those baselines **on our datasets under our protocol**, using their official code where available. If time permits, we additionally reproduce one baseline on its own dataset as a sanity check of the re-implementation.

## 5. Pre-processing rules (leakage-safe)

1. **No global k-core.** The official 5-core files filter on the *entire* timeline, so a user's future interactions decide whether their past is kept. That is temporal leakage, and it also deletes the cold entities we need. We apply k-core filtering (k=5 for users and items) **only within the training-period data available at each step**, and evaluate on all test interactions whose user or item is either known or *new* (new entities go to the open-world track).
2. **Implicit feedback.** Rating ≥ 4 → positive (main setting). As a robustness check, also report "all reviews as positives".
3. **De-duplicate** on (user, parent_asin), keeping the earliest timestamp.
4. **Item text.** Title + categories + the first 512 tokens of features/description. It is encoded once with a frozen sentence encoder. The encoder must be published before the data window ends, or else the encoder choice must be stated as a contamination risk (see `experimental_design.md` §10).

## 6. Chronological splits

### 6.1 Blocks

* Partition the window into quarterly blocks $B_1, \dots, B_T$.
* **Warm-up**: $B_1..B_{W_0}$ with $W_0 = 8$ (2 years), used for initial training.
* **Stream**: $B_{W_0+1}..B_T$, about 27 blocks.

### 6.2 Prequential protocol (main)

For each $k \ge W_0$:

1. **Test (OOD):** evaluate the model trained up to $B_k$ on $B_{k+1}$ *before* any update with $B_{k+1}$. This gives next-block accuracy, calibration and selective risk.
2. **Validation:** the last 10% of $B_k$ by time, held out *during* the update on $B_k$. It is used for early stopping and for fitting the calibration parameters. No hyperparameter is tuned on any test block.
3. **Update** on $B_{k+1}$ (minus its last 10%, which becomes the validation slice for the next step).
4. **Retention:** after the update at step $k$, also evaluate on held-out test slices of earlier blocks $B_j$, $j \le k$, to compute forgetting and BWT.

### 6.3 Hyperparameter tuning

* Tune **only** on the warm-up period: train on $B_1..B_{W_0-2}$ and validate prequentially on $B_{W_0-1}, B_{W_0}$.
* Freeze the hyperparameters for the whole stream, with the same budget for every method (e.g., 30 trials of TPE or a fixed grid).

### 6.4 Entity tracks in each test block

* **Warm:** user and item seen before $B_{k+1}$.
* **New-user:** user first appears in $B_{k+1}$. Evaluate after revealing their first $n \in \{0,1,3,5\}$ interactions, ordered by time, and rank the remainder.
* **New-item:** item first appears in $B_{k+1}$.

### 6.5 Controlled synthetic-drift variant (Video_Games)

At a chosen block, for a random 10% of users with ≥ 20 interactions, permute their *future* interactions to a different item category cluster. This is a known ground-truth preference change. It is used only for detector evaluation (H3) and the stability–plasticity analysis (H4), and is clearly labelled synthetic.

### 6.6 Optional comparability split

The official Amazon-2023 absolute-timestamp split (t1 ≈ 2021-08-11, t2 ≈ 2022-07-16 UTC, converted from the official ms values) can be reported once as a single-shift static-OOD comparison against DR-GNN / DRGO-style evaluation.
