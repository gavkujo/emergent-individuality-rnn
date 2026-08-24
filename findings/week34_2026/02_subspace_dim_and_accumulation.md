# The idle limit cycle is 2D, and B3 accumulation doesn't survive multi-seed

**Date**: 2026-08-22
**References**:
  - `results/week34_2026/subspace_dim_20260822_v1.json`
  - `figures/subspace_dim/*.png` (4 figures)
  - Code: `repo/src/experiments/subspace_dim/`

## Two findings from one experiment

Ran the current benchmark's wake-sleep training on 3 seeds × 3 streams
(A_low, B_high, D_reversed) and measured (a) the effective dimensionality
of the resulting idle trajectories, and (b) the n-sensitivity of the
subspace-angle-based metrics the paper reports.

## Finding 1: idle trajectories are 2D

Across every seed × stream (9 configs total):

| Threshold | n needed |
|---|---|
| 90% variance | 2 in every config |
| 95% variance | 2 in every config |
| 99% variance | 2 in 4 configs, 3 in 3 configs, 4 in 2 configs |

Typical top-3 PC variance shares: **60% / 40% / 0.5%**. Everything past
PC2 is < 1% of the trajectory's variance. The idle limit cycle is
essentially 2D — a curve in state space, as one would expect for a
sinusoidal-trained network.

Current `subspace_angle` uses `n=3`. This includes one noise PC in the
comparison. The reported angle averages over one real principal-angle
and two noise principal-angles, and depends non-trivially on how the
noise directions happen to align across seeds.

**Recommendation**: use `n=2` as the principled choice. This is
justified by the 95%-and-99%-variance analysis and is architecture-
independent (it just says "compare the actual limit-cycle plane, not
the noise").

## Finding 2: B3 accumulation is not robust across seeds

Cycle-1 vs cycle-5 subspace angle (A_low vs B_high), per seed, at
n=3 (paper's metric):

| Seed | c1     | c2     | c3     | c4     | c5     | Δ (c5 − c1) |
|---|---|---|---|---|---|---|
| 42   | 56.17° | 66.40° | 62.78° | 59.42° | 58.87° | **+2.70°**  |
| 43   | 57.66° | 68.08° | 57.91° | 64.55° | 55.45° | **−2.20°**  |
| 44   | 62.93° | 57.77° | 61.27° | 59.77° | 54.75° | **−8.18°**  |

At n=3, **2 of 3 seeds show a DECREASE** in angle over 5 wake-sleep cycles.
Mean slope across seeds: **−0.795 °/cycle**. The paper's B3 claim of
"subspace angle 54° → 63° over five cycles" was a single-seed
observation that does not survive multi-seed measurement.

At n=2 (principled), the direction is on average slightly positive
(mean slope +0.21°/cycle) but with high seed variance:

| Seed | Δ over 5 cycles at n=2 |
|---|---|
| 42 | +10.26° |
| 43 | −5.70°  |
| 44 | +4.69°  |

Two of three seeds positive, mean about +3 °/cycle but with 6 °/cycle
seed-to-seed variance. Not a compelling accumulation claim from n=3.

**This is the more important of the two findings.** The paper's
narrative that "divergence accumulates across wake-sleep cycles" is
an artefact of measuring one seed with the wrong n. With 3 seeds and
either n=2 or n=3, there is no reliable monotonic accumulation.

## Finding 2b: B6 order effect DOES survive

The B6 claim (same tasks reversed produces less divergence than
different tasks) holds robustly across seeds at both n=2 and n=3:

| Seed | ratio (A-vs-D) / (A-vs-B), n=2 | n=3   |
|---|---|---|
| 42   | 0.825                          | 0.886 |
| 43   | 0.827                          | 0.950 |
| 44   | 0.800                          | 0.880 |
| mean | **0.817**                      | 0.905 |

All three seeds < 1 in the "right" direction at both n. The paper's
"86%" number came from n=3 and a specific seed; the true multi-seed
mean at n=3 is 0.91, and at n=2 it is 0.82. The claim survives
qualitatively.

At n=1 the ratio is *not* robust (seed 42 gives 1.52, seed 43 gives
0.85, seed 44 gives 0.77) — the single dominant PC is not enough
representational capacity to pick up the order structure. n ≥ 2 is
required and n=2 is defensibly principled.

## What this means

Three consequences for the workshop paper.

### 1. Change the subspace_angle default to n=2

Everywhere `subspace_angle` is called, the default should be n=2. The
current n=3 has no theoretical justification and inflates the metric
against a noise dimension. All benchmarks that use it should be re-run
with n=2 for the paper. This is a small code change with substantial
downstream effects.

### 2. Drop the B3 accumulation claim, at least until multi-seed

The paper's "54° → 63° accumulation" story is single-seed and does not
generalise. Two of three seeds actually show angle DECREASING at n=3,
and even at n=2 the seed variance dominates the mean effect.

The claim needs to be either dropped or rewritten as: "cycle-to-cycle
divergence between two histories varies substantially seed-to-seed;
we do not observe a consistent monotonic accumulation trend." This is
a real finding — the paper cannot present accumulation as a headline
result at n=1 seed.

The natural follow-up is multi-seed (n≥10) at n=2 and n=3, to see
whether the mean slope is statistically distinguishable from zero.
This is on `todo_hardening.md` item 7.

### 3. B6 order effect and B2 decoder accuracy remain load-bearing

The two claims that DO survive proper methodology:

- **B2** decoder accuracy at held-out 5-fold CV = 1.000 ± 0.000
  (finding 01_decoder_holdout not yet written, but the numbers are in
  benchmark v3 JSON)
- **B6** order effect ratio ≈ 0.82 at n=2 across 3 seeds (< 1 in every
  seed)

These are what the workshop paper should be built around. The
architecture-specific richness claim died in finding 02 (week 22), the
accumulation claim dies here, the decoder and order-effect claims
remain.

## Caveats

- Only 3 seeds. Multi-seed at n≥10 is item 7 in todo_hardening.md and
  is needed for statistical claims. This finding is directional, not
  publication-quality; it establishes that the accumulation-story
  problem is real, not that a specific slope estimate is correct.
- One task family (sinusoids). Chirps/pulse-trains/Lorenz replication
  is item 6.
- Idle length = 300 steps. Different idle lengths might reveal
  different effective dim (though we expect the intrinsic dim of a
  limit cycle to be robust to observation length).
- `subspace_angle` averages principal angles. At n=2 we average two
  angles. An alternative is to take the largest principal angle (the
  "worst-case" divergence direction). This gives different numbers
  and may be more informative for the B3/B6 story. Not measured here.

## What changes in the codebase

- `subspace_angle(..., n=3)` default → **n=2** (patch on `src/train.py`).
- benchmark v3 → v4 when rerun with new default (or add a note that
  new numbers are n=2 while old JSON was n=3).
- `paper.md` accumulation section rewritten or removed.

**These are code changes, not done in this experiment.** They are the
concrete follow-up. This finding documents the reason for the change.

## Open question

Is the B3 pattern *actually* absent, or is it present but too small to
see with 3 seeds? With 10 seeds we'd expect standard error ≈
(seed_std) / √10 ≈ 6 / 3 ≈ 2 °/cycle. If the true mean slope is 0.5
°/cycle, we'd still fail to detect it. If the true slope is > 3
°/cycle, we would.

This is a real design question: at what number of seeds is the
accumulation claim testable? If we need 100 seeds to reach adequate
power, the effect is too small to be a headline. If 10 suffice, we
can decide.
