# Order effect at n=30 with recency neutralised: middle order matters, less than middle content

**Date**: 2026-08-24
**References**:
  - `results/week35_2026/order_effect_anchored_20260824_v1.json`
  - Code: `repo/src/experiments/order_effect_anchored/`
  - Prior findings this resolves:
    - `05_order_effect_hardened_v1.md` (v1 confounded by recency)
    - `07_order_effect_hardened_v2_null.md` (v2 null; couldn't
      distinguish recency-dominance from equal-contribution)

## The setup

Anchored version of the B6 order test. All three streams share the SAME
last task; only the first K-1 = 4 cycles vary. This isolates
middle-of-stream effects from recency.

- **n = 30** seeds. Wake+sleep training, canonical config.
- Per seed:
  - `anchor` = one freq drawn log-uniformly from [0.5, 20] Hz.
  - `base_freqs`  = 4 freqs, log-uniform. Middle content for A and B.
  - `alt_freqs`   = 4 fresh freqs, log-uniform. Middle content for T.
  - `perm_A`, `perm_B` = two independent random permutations of 4 slots.
  - `S_A` = base_freqs in perm_A, then anchor.
  - `S_B` = base_freqs in perm_B, then anchor.
  - `T`   = alt_freqs  in perm_A, then anchor.
- All three networks per seed start from the SAME
  `torch.manual_seed(seed)` init.

## Results

### Absolute distances (all streams end at identical anchor task)

| Comparison | angle n=2 | angle n=3 | L2 |
|---|---|---|---|
| S_A vs S_B  (same content, DIFFERENT middle order, same last)    | **64.74°** | **66.70°** | 6.88 |
| S_A vs T    (DIFFERENT middle content, same middle-slot order, same last) | **71.87°** | **73.49°** | 7.96 |

95% CIs:

| Comparison | angle n=2 CI | angle n=3 CI | L2 CI |
|---|---|---|---|
| S_A vs S_B | [59.4, 70.1] | [62.2, 70.6] | [5.75, 8.12] |
| S_A vs T   | [68.1, 75.4] | [70.6, 76.0] | [6.78, 9.17] |

### Ratios middle-order / middle-content

| Metric | Ratio | 95% CI | Seeds with ratio < 1 |
|---|---|---|---|
| angle n=2 | **0.909** | [0.842, 0.966] | 22 / 30 |
| angle n=3 | **0.915** | [0.852, 0.975] | 22 / 30 |
| L2        | 0.922 | [0.798, 1.043] | 15 / 30 |

The angle ratios have CIs that EXCLUDE 1 on the low side. So
middle-order and middle-content are not statistically equal — content
edges out order.

### Comparison to v2 (non-anchored)

| Metric | anchored A vs B | v2 A vs B | anchored A vs T | v2 A vs T |
|---|---|---|---|---|
| angle n=2 | 64.7° | 72.0° | 71.9° | 74.5° |
| angle n=3 | 66.7° | 73.2° | 73.5° | 74.3° |

Anchoring drops the "same content, different order" angle by ~7° and
the "different content, same order" angle by ~2°. Recency contributes
a small but measurable slice of the total divergence — but middle
experience carries the bulk of it.

## Interpretation

**Middle-order matters.** Two networks trained on the same 4 tasks in
different order, ending at the identical anchor task, still develop
64.7° subspace angles at n=30. This is nearly the same as networks
trained on entirely different histories. The paper's "order shapes
identity" claim is defensible.

**Middle-content matters slightly more.** Ratio 0.91 with CI [0.84,
0.97] — content variation produces ~10% more divergence than order
variation. Same direction as the paper's original B6 result (0.86 at
n=1), holding at n=30 with recency neutralised.

**Recency contributes a small amount.** ~7° for order and ~2° for
content. Not negligible but not the dominant source.

**Individuality is high-dimensional.** Even with the last task fixed
and the specific freqs fixed (only the ORDER varying), networks land
in nearly-orthogonal state-space regions (~65° out of 90°). Wake+sleep
training is highly sensitive to the sequence of experience.

## Combined picture (findings 03, 04, 05, 06, 07, 08)

**Confirmed at n=30 with proper controls (paper claims that survive):**

- Networks trained on different experiential histories develop
  distinguishable idle-state geometries. Effect is large (~65-75°
  subspace angle) and consistent across 30 seeds.
- The signature is decodable within-seed at K=20 (ridge = 1.000
  across all 30 seeds).
- Order of experience is a genuine dimension of individuality,
  producing 91% of the divergence that content variation produces
  (CI [84%, 97%], excludes 1).
- Middle-of-stream experience carries the bulk of the divergence;
  recency contributes a small (~2-7°) additive slice on top.
- The signature is not attributable to weight-norm or
  mean-activation confounds.

**Refuted at n=30 (paper claims that must be dropped or reframed):**

- **B3 accumulation cycle-by-cycle** — dropped (finding 03).
- **B5 sleep amplifies divergence at L=5** — inconclusive, CI covers
  zero (finding 06). Possible at longer L but not shown.
- **Trajectory geometry / temporal shape encodes history** — refuted
  by the shuffled=main=1.000 result (finding 04). The decoder reads
  per-timestep marginals, not trajectory shape.
- **Universal task-to-state signature across networks** — refuted by
  the pooled cross-seed mean-state decoder at chance (finding 04).
  The signature is idiosyncratic per random init.

## Impact on the paper

The paper's spine is defensible at n=30. What changes vs the current
draft:

- **B2**: keep as headline (1.000 at K=20, n=30). Add the two null
  controls: shuffled = main (drop the trajectory-shape language) and
  cross-seed pooled = chance (reframe as "each network has its own
  task-specific fingerprint" rather than "task history has a canonical
  signature").
- **B3**: DROP the accumulation claim entirely. Optional footnote:
  "we do not observe monotonic accumulation of divergence over cycles
  at n=10 seeds."
- **B4**: RUN a hardened version before submitting. Not blocked, but
  the number needs a CI.
- **B5**: DROP "sleep amplifies divergence" (n=30 CI covers zero at
  L=5). Retain "sleep consolidates the limit cycle" as a mechanism
  claim without a divergence number attached.
- **B6**: KEEP the order effect claim. Reword to "order of experience
  produces 91% of the divergence that content variation does, with
  95% CI [84%, 97%], anchored at the same terminal task, at n=30."
  Cite the anchored experiment methodology.
- Add a section reporting the shuffled-time control and the cross-seed
  mean-state control. Frame these as *strengthening* the case (both
  potential confounds ruled out), while forcing the honest reframing
  above.

## What remains before submitting

- **Task-family replication** (Phase D). The claims above are all on
  sinusoids. NeurIPS reviewer will ask whether it holds on chirps /
  pulse trains / Lorenz. At least one non-sinusoid family should
  replicate before submission.
- **Fair-baseline table** (Phase E). If we claim LeakyRNN develops
  individuality, we should compare against VanillaRNN, GRU, LSTM,
  FrozenESN with fair input pathways.
- **Path-dependence at n=30** (`path_dependence_hardened`). The paper's
  B4 claim needs a CI.
- **Continual-learning baseline (EWC / replay)** if the paper is going
  to distinguish itself from continual-learning literature.

None of these are blockers for a workshop paper if scope is honest.
All are blockers if we want the paper to survive a NeurIPS main-track
review later.
