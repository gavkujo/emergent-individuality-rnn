# Order effect at n=30 with recency neutralised: no distinguishable effect

**Date**: 2026-08-24
**References**:
  - `results/week35_2026/order_effect_hardened_v2_20260824_v2.json`
  - Code: `repo/src/experiments/order_effect_hardened_v2/`
  - Prior findings:
    - `05_order_effect_hardened_v1.md` (v1 confounded by recency)
    - `04_decoder_hardened_K20_n30.md` (decoder controls)

## The setup

`order_effect_hardened_v2` fixes v1's recency confound. Per-seed
streams:

- Sample K=5 base freqs (log-uniform [0.5, 20]) once per seed.
- **`S_perm_A`**: random permutation of these 5.
- **`S_perm_B`**: another random permutation of the SAME 5.
- **`T_perm`**:   random permutation of 5 FRESH freqs from the same
  distribution.

Same wake+sleep training, 30 seeds, canonical hyperparameters.

Recency-neutralisation diagnostic (mean of |log(last_A) − log(last_x)|
across 30 seeds): AB = 1.22, AT = 1.29, difference = **−0.07**.
The last-freq distributions are matched. This is a clean test.

## Results

### Absolute distances

| Comparison | angle n=2 | angle n=3 | L2 |
|---|---|---|---|
| A vs B (same content, different order) | **72.0°** | **73.2°** | 7.21 |
| A vs T (different content, matched distribution) | **74.5°** | **74.3°** | 7.04 |

### Ratios

| Metric | Ratio (mean of per-seed) | 95% CI | Seeds with ratio < 1 | Range |
|---|---|---|---|---|
| angle n=2 | **1.007** | [0.905, 1.119] | 14 / 30 | [0.39, 2.08] |
| angle n=3 | **1.000** | [0.929, 1.068] | 15 / 30 | [0.49, 1.61] |
| L2        | **1.100** | [0.930, 1.279] | 13 / 30 | [0.34, 2.22] |

Every CI includes 1. Sign counts are at chance. The per-seed range is
enormous (some seeds have ratio 0.34, others 2.22).

## Verdict

At n=30 with recency neutralised, the order of the same K tasks and a
resample of K new tasks from the same distribution produce
**statistically indistinguishable** amounts of divergence. The
paper's B6 "order matters as a first-class dimension of individuality"
claim does not survive.

## What this actually says about individuality

Look at the absolute magnitudes: 72–74° angles out of a max of ~90°.
Both same-content-reordered and different-content-resampled produce
idle-state subspaces that are effectively orthogonal to the reference.

**Any variation in wake experience — order OR content — sends the
network into a nearly-uncorrelated corner of idle-state space.**

Two ways to read this:

1. **High-variance idiosyncrasy.** Each network is so seed-dependent
   and so experience-sensitive that any deviation in training data
   moves it a large distance in state space, regardless of the
   nature of the deviation.
2. **Recency dominance** (from v1) plus content-variation-produces-
   equally-large-recency-effects. Under random permutation, the last
   task of A is uncorrelated with the last task of B (both random draws
   from the same distribution); the last task of A is also uncorrelated
   with the last task of T (same reason). So both AB and AT distances
   are determined by "last-task gap" plus "everything else is noise."

Both are consistent with the data. The second is more parsimonious
given the wake-only decoder result (shuffled = main = 1.000; per-
timestep values alone are class-decodable).

## Implications for the paper (combined view of experiments 04, 05, 07)

The three hardened multi-seed experiments together tell a coherent story:

**Confirmed at n=30, multi-seed with CIs:**

- Wake training produces idle states linearly separable at K=20
  (ridge = 1.000, all 30 seeds).
- The fingerprint is not attributable to weight-norm or
  mean-activation confounds.
- Any variation in experience produces idle-state divergence of
  substantial magnitude (~73° subspace angle over 5 wake-sleep cycles).

**Refuted at n=30:**

- History accumulates cycle-by-cycle (B3, `03_accumulation_...`).
- Sleep amplifies divergence at L=5 (B5, `06_sleep_effect_...`).
- Trajectory shape (temporal structure of the idle limit cycle)
  encodes history (`04_decoder_hardened_...`, shuffled = main).
- The task → idle-state mapping is universal across networks
  (`04_decoder_hardened_...`, pooled mean-state = chance).
- Order is a first-class dimension of individuality distinct from
  content (this finding).

**What's left of the "individuality" claim** — the honest workshop
paper thesis — is:

> Networks trained on different experiences develop idiosyncratic
> idle-state geometries. Within a network, task-specific fingerprints
> are highly linearly decodable. Across networks, the same task
> produces uncorrelated fingerprints. The fingerprint is not
> attributable to trivial confounds, is not encoded in temporal
> trajectory shape, and is not distinguishable from a "recency-
> dominated pattern generator" hypothesis with existing experiments.

This is a much smaller claim than the paper currently makes, but it
is defensible at n=30 with proper controls.

## Where "order matters" could still be rescued

Only if a follow-up experiment isolates something the v2 design
does not:

- **`order_effect_anchored`**. Force `S_A` and `S_B` to have the SAME
  last-task freq (or the SAME last k tasks), permute only the earlier
  cycles. If divergence drops to a noise floor (small angle), the
  entire signal is recency and order is not a mechanism. If a
  substantial angle remains (say > 30°), earlier order matters and
  the paper can defend an order claim.

- **Longer streams (L = 20 or 30)**. At L=5, the accumulated
  contribution of "history beyond the last cycle" is dominated by
  the last cycle's residue. At L=30, if the sleep phase's Hebbian
  contribution is real, its integrated effect across 30 cycles
  should produce a signature independent of the last cycle. Same
  ratio test at L=30 would either confirm or bury "order matters."

Neither test is done. Until one is, the honest position for the
workshop paper is: order effect NOT supported.

## What to do next

Options, in priority:

1. **`order_effect_anchored`**. Cheap (same cost as v1/v2). Clean
   test of "recency vs order-beyond-recency." If a substantial
   anchored angle survives at n=30, the order claim can be resurrected
   in a different form. If not, we know for sure.

2. **`decoder_stream_hardened`**. The paper's OTHER decoder variant
   uses full 5-cycle streams, not single tasks. Does the cross-seed
   mean-state pool still fail when the "class" is a full history
   rather than a single task? This tests whether pooled generalisation
   emerges when experience is richer.

3. **Sleep effect at L=30**. Only after option 1 clarifies the
   recency question. If order-beyond-recency exists, sleep at L=30
   can amplify it. If it doesn't, L=30 sleep is unlikely to help.

4. **Paper rewrite (Phase F)**. Enough data to draft an honest paper
   spine. But probably wait for one of the above to firm up.

## Impact on the paper (working list)

**Section-by-section rewrite plan** (informal, not yet a checklist item):

- Abstract: drop "accumulates," drop "order matters," drop
  "trajectory geometry." Keep "networks develop task-specific
  fingerprints, idiosyncratic across inits."
- B2 decoder: keep the K=20 result with CI. Add controls. Add
  cross-seed pool caveat.
- B3 accumulation: drop (already flagged).
- B4 path-dependence: needs its own hardened rerun before we know.
- B5 sleep effect: report as null with CI, not "10% amplification."
- B6 order effect: drop the "order matters" claim. Reframe as
  "content variation and order variation produce equivalently large
  divergences" or defer to `order_effect_anchored` for a stronger
  version of the claim.
- Sleep-mechanism story: reframe from "amplifies divergence" to
  "consolidates the limit cycle" — a separate, non-quantitative
  claim.
