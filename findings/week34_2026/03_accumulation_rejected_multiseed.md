# B3 accumulation is not supported at n=10 seeds

**Date**: 2026-08-23
**References**:
  - `results/week34_2026/accumulation_multiseed_20260823_v1.json`
  - `figures/accumulation_multiseed/*.png` (3 figures)
  - Code: `repo/src/experiments/accumulation_multiseed/`
  - Prior finding this replicates:
    `findings/week34_2026/02_subspace_dim_and_accumulation.md`

## The result

Re-ran the workshop paper's B3 experiment (A_low vs B_high, 5 wake-sleep
cycles, cycle-by-cycle divergence) at **n = 10 independent seeds** (torch
seeds 42..51), same architecture and hyperparameters as the paper.
Measured three metrics per cycle: subspace angle at n=2 (post-hardening
default), subspace angle at n=3 (legacy paper setting), and L2
divergence. Fit a linear slope (metric ~ cycle) per bootstrap resample of
seeds (2000 resamples) to get a 95% CI on the population slope.

| Metric | Mean-curve slope | 95% CI | Seeds with positive Δ (c5 − c1) |
|---|---|---|---|
| subspace angle, n=2  | **+0.30°/cycle** | [−0.56, +1.19] | 6 / 10 |
| subspace angle, n=3  | **−0.09°/cycle** | [−0.70, +0.61] | 5 / 10 |
| L2 divergence        | **−0.037 /cycle** | [−0.071, −0.001] | 5 / 10 |

**Verdict on each metric:**

- Subspace angle at n=2: **inconclusive.** The CI on the slope covers
  zero, and 4 of 10 seeds go the wrong way. Even the point estimate
  (+0.3 °/cycle) implies an expected c1→c5 growth of ~1.5°, well within
  seed-to-seed variance.
- Subspace angle at n=3: **inconclusive, leaning negative.** Point
  estimate near zero, CI covers zero.
- L2 divergence: **weakly rejected.** The CI just barely excludes zero on
  the negative side (upper bound −0.001), meaning across seeds the L2
  metric on average *slightly decreases* over cycles, opposite to the
  paper's claim.

## Comparison to the workshop paper's original numbers

The workshop draft reports B3 as:

> "Subspace angle between Stream A and Stream B: 54.47° after cycle 1,
> 62.17° after cycle 5. L2 distance: 4.431 → 4.405."

This was **one seed**. At n = 10 seeds, the equivalent per-cycle averages
at n = 3 are 59.9° → 59.3°: no monotone growth, and the c1 vs c5 gap
(−0.6°) is much smaller than any single seed's Δ (min −8.2°, max +7.6°).

The paper's specific numbers do not survive multi-seed replication. The
paper cannot report "54° → 63° accumulation" as a general property of the
architecture.

## Individual seed variance

Per-seed Δ = angle(cycle 5) − angle(cycle 1), at n = 2:

```
+10.3, −5.7, +4.7, +9.0, −1.5, −4.3, +2.3, +6.6, −6.5, +3.9
```

The spread (min −6.5°, max +10.3°) is much larger than the mean (+1.7°).
Any single seed can produce a strongly positive or strongly negative
"accumulation" number. The workshop paper picked one of the positive
ones by chance.

## What this means

### For the workshop paper

The B3 "divergence accumulation" claim, as currently stated, is not
supported. Three responses in decreasing order of ambition:

1. **Drop the claim.** Rewrite the B3 section as: "cycle-to-cycle
   divergence between two histories varies substantially seed-to-seed;
   we do not observe a consistent monotonic accumulation trend."
   This is the honest reading of the data and the cleanest path.

2. **Reframe as "path-dependent stability."** The variance in per-seed
   trajectories itself is a finding: the two networks end up at
   different distances in different runs, but they end up at *some*
   distance every time, and that distance is decodable (B2) and
   order-sensitive (B6). Frame B3 as evidence of stable inter-network
   distance, not growing distance.

3. **Keep chasing accumulation with more compute.** Increase seed count
   to 30 or 50 with parallel training. At 30 seeds the standard error
   drops to about √3 = 1.7× tighter; the CI on the n=2 slope would
   likely still include 0 given the current point estimate of +0.3
   °/cycle and per-seed std of ~6°.

I recommend option 1 for the workshop, option 2 as an alternative
framing to consider, option 3 for the FYP main track only if
accumulation matters for Checkpoint 4.

### For the FYP main track

Checkpoint 4 (Self-Reinforcing Individuality, per
`other_docs/fyp_scope.md`) is built on the premise that individuality
deepens across experience. This finding does not directly disprove
Checkpoint 4 — that checkpoint operates over "10+ tasks" of *distinct*
learning, not repeated wake-sleep on the same stream. But the shape of
the concern is the same: is individuality a growing quantity, or a
stable-once-established fingerprint?

The Path B contingency in `todo_hardening.md` is now the primary path
to consider for the FYP main track, not a fallback. The paper's headline
becomes:

  "Identity emerges from experience and remains stable and decodable;
  its shape depends on the order of experiences. We find no evidence
  that identity intensifies monotonically across additional wake-sleep
  cycles on similar experiences."

This is more careful and probably more accurate.

### For the current codebase

- `src/experiments/benchmark/__init__.py` produces B3 numbers that
  are single-seed and misleading. Either delete B3 from benchmark, or
  keep it and note in the DETAILS that the multi-seed replication in
  `accumulation_multiseed` contradicts the accumulation reading. I
  recommend the latter — the per-seed numbers themselves are useful
  as raw data.
- The 3-point sanity sweep (todo item 9) should not use B3 as one of
  its "primary" load-bearing metrics. B2 (decoder) and B6 (order
  effect) are the load-bearing metrics that survived multi-seed
  measurement.

## Caveats

- **Single task family (sinusoids).** The multi-seed result rejects
  accumulation *for sinusoidal stream training*. Chirps, pulse trains,
  and Lorenz have not been tested (todo item 7). It's possible that
  richer task families produce genuine accumulation. This is not
  currently known.
- **Single stream pair (A vs B).** The workshop paper's B4 experiment
  varies stream length while keeping A vs B; that experiment has not
  been multi-seeded. It is unclear whether B4 replicates or shows the
  same seed-variance-dominated pattern.
- **5 cycles.** Long-training behaviour (20+ cycles) may look
  qualitatively different. Not measured.
- **Bootstrap CI at 2000 resamples with n = 10 seeds** is at the edge
  of what the sample supports; the CIs here are directional not razor-
  sharp. But even at n = 100 seeds, given the observed per-seed
  std ~ 6° and mean slope ~ 0.3°, we'd expect the CI to still contain
  zero unless the true slope is at least ~1.5 °/cycle.
- **Bootstrap resamples the slopes of the mean curve, not per-seed
  slopes directly.** Both should give similar answers; the mean-curve
  approach is slightly more optimistic. Per-seed slopes have wider
  variance.

## Open question

Does anything grow with cycles? We haven't measured other quantities:
per-network variance of the trained W_rec, dominant Lyapunov exponent
of the idle dynamics, effective attractor dimension, decoder confidence.
Any of these could accumulate even if pairwise divergence does not.
Worth testing on the FYP main track if the "growing individuality"
framing is to be recovered.
