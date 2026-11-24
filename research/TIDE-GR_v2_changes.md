# TIDE-GR v2 — what went wrong in v1 and what v2 changes

Branch `v2`. v1 stays unchanged on `main`; v2 writes to its own Drive folder (`TIDE_GR_v2_runs`) and Sheet.

## 1. v1 results that v2 must fix (KuaiRand-Pure, 17 stream days)

| | NDCG@20 |
|---|---|
| SASRec (3 seeds) | 0.0720 |
| TIDE-GR v1 (3 seeds) | 0.0662 |
| v1 with RL policy / random arms / no RL | 0.0662 / 0.0662 / 0.0662 |
| v1 without replay memory | 0.0696 |

The RL controller had no effect, and the replay memory hurt accuracy *and* retention.

## 2. Why the v1 RL controller failed (code + log evidence)

1. **The reward could not be learned.** The reward was the next-day NDCG of each user group, centred
   across the 5 groups of the same day. The groups are drift quintiles, so they differ in difficulty
   regardless of the arm. The centred reward measured "which group is easier", not "which arm is better".
2. **Wrong credit assignment.** Each group got its own arm, but only `q1` acted per group. Epochs and
   replay (the levers with real effect) were taken from the *majority* arm of the day and applied to
   everybody. A group's reward was then credited to an arm it did not actually receive.
3. **The arms had almost no leverage.** `epoch_mult` changed little, because early stopping (patience 3)
   decides the epochs anyway. `q1` only changes the anchor precision a little. The lever that matters
   (how much of the past is replayed) was never varied in a useful way. The `a4_skip` arm skipped a whole
   day of learning, a costly mistake in a test-then-train stream.
4. **Too little data for the prior.** There were 17 days × 5 groups of noisy rewards for 5 arms × 8 weights.
   The prior (std 0.01) plus the +0.02 bonus for arm 0 kept the policy at `a0_default` most of the time.
   That is why RL, random arms and no RL all gave 0.0662.

## 3. Other problems found

* **Replay memory (the main accuracy loss).** Each update replayed as many old edges as new ones, sampled
  uniformly from the pooled memory. The memory slices of the big early days (up to 93k edges/day, versus
  3–8k per stream day) dominated, so every update pulled the model back to the old days. In the history
  check below, removing or shrinking it gave the largest gain, and it also *improved* retention.
* **Stream optimisation (checked, not a cause).** Stream days have only 3.4k–7.7k training edges, so the
  batch of 4096 gives 1–2 optimiser steps per epoch. A batch of 512 did *not* help (0.0715 vs 0.0717).
  The option `stream_opt` stays, off by default.
* **Recency weights in the recent-interest term (checked, no gain).** Option `recent.decay`, off by default.
* **Faster edge ageing (checked, no gain).** Not used.
* **Tuning wasted trials.** A trial equal to the default was trained twice (fixed in `tune.py`).
* **EWC / KD-lite grids too narrow.** They behaved like plain fine-tuning; the grids are now wider, so a
  reviewer cannot call these baselines under-tuned.

## 4. What v2 changes

1. **Recency-weighted, small replay** (`memory.sampling: env_recency`, `recency_decay: 0.7`). Every stored
   day gets weight 0.7^(age), and edges are uniform inside a day. The amount replayed is chosen by the RL arm.
2. **RL controller rebuilt, and kept as the core part** (`rl.mode: shadow`, `rl_bandit.PairedTSBandit`):
   * **Paired counterfactual reward.** Every arm is trained from the *same* state as a shadow model. The
     next day, all of them are scored on the *same users*. Reward = (NDCG of arm) − (NDCG of the reference
     arm), plus 0.25 × the same difference on the memory-validation slices (stability; never the
     retention test slices). Day difficulty and group difficulty cancel exactly. This fixes causes 1 and 2.
   * **Arms on real levers.** Replay amount (0, 0.25, 1.0 per new edge), anchor strength (×0.1 to ×3),
     forgetting rate q1, and step size. There is no skip arm. This fixes cause 3.
   * **Full feedback.** Every arm is observed every day (shadows), so learning needs no exploration
     cost. Thompson sampling chooses an arm only when its sampled gain over the reference is > 0 (safe
     fallback to the reference arm). This fixes cause 4.
   * The reference arm `a0_default` equals the best fixed setting found on history (0.25 recency replay).
3. Tuning grid and plan: history-only coordinate search over the remaining v1 boosters. The ablations test
   every RL claim: fixed arms, random arms, bandit feedback only, no drift context, accuracy-only reward,
   v1-style replay.

## 5. History-only evidence used to choose the v2 defaults

Pseudo-stream on the *history* days 6–12 (warm-up on days 0–5). The stream test days were not used.
Seed 0, mean NDCG@20 over days 6–12.

| Variant | NDCG@20 | Retention R@20 | BWT |
|---|---|---|---|
| v1 | 0.0717 | 0.1757 | −0.0196 |
| v1 + batch 512 | 0.0715 | 0.1743 | — |
| **v1 + recency replay 0.25** | **0.0736 (+2.6%)** | **0.1864** | **−0.0169** |
| v1 + no replay | 0.0742 (+3.5%) | 0.1841 | −0.0262 |
| + anchor ×0.3 | 0.0733 | 0.1860 | −0.0223 |
| + faster edge ageing | 0.0735 | 0.1845 | −0.0209 |
| + recent decay 0.8 | 0.0734 | 0.1812 | −0.0187 |

Replay is a real stability–plasticity lever. No replay gives the best accuracy but the worst backward
transfer. Recency replay gives the best retention and BWT. Which is better changes from day to day,
which is exactly the decision the RL controller now makes.

A local re-run of v1 on the full stream gave 0.0672 (Colab seed 0: 0.0662), so the pipeline reproduces.

## 6. Honest limits

* The gains above are single-seed history results. The real numbers come from the v2 plan (3 seeds +
  ablations). v2 is expected to close part of the gap to SASRec. It is **not** guaranteed to beat it.
* v2 was designed after seeing the v1 stream results. The paper must say so. The v2 defaults themselves
  were chosen on history days only.
* The shadow controller trains every arm every day (4 arms ⇒ about 3–4× the update cost of v1; on a T4
  this is still only a few seconds per day).
