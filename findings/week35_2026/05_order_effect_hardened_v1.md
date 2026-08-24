# Order-effect ratio at n=30: 1.70 (opposite direction of paper's B6), but confounded by recency

**Date**: 2026-08-24
**References**:
  - `results/week35_2026/order_effect_hardened_20260824_v1.json`
  - Code: `repo/src/experiments/order_effect_hardened/`
  - Prior finding this contrasts with:
    `findings/week35_2026/04_decoder_hardened_K20_n30.md`

## The setup

Industry-standard replication of the paper's B6 order-effect claim.

- **n = 30** independent random inits (torch seeds 0..29).
- **Stream length L = 5** wake-sleep cycles, matching the paper.
- Per-seed sinusoid streams, log-uniformly-sampled frequencies from
  [0.5, 20] Hz. Two independent streams per seed:
  - `S_fwd`: 5 freqs, one from each of 5 log-spaced bins, ordered LOW → HIGH.
  - `S_rev`: `reverse(S_fwd)`, ordered HIGH → LOW. Same tasks, opposite order.
  - `T`:    5 fresh freqs, independently sampled from the same distribution.
- All three networks per seed start from the SAME
  `torch.manual_seed(seed)` init. Wake+sleep training, all canonical
  hyperparameters. Idle 400 steps.
- Divergence measured three ways: subspace angle at n=2, n=3, and L2.
- Ratio = angle(S_fwd, S_rev) / angle(S_fwd, T). Bootstrap 95% CI over
  30 per-seed ratios, 2000 resamples.

## Results

### Absolute distances

| Comparison | angle n=2 | angle n=3 | L2 |
|---|---|---|---|
| `S_fwd` vs `S_rev`   (same content, reversed order) | 74.8° | 77.4° | 7.34 |
| `S_fwd` vs `T`       (different content, same style) | 45.2° | 52.9° | 4.62 |

### Ratios (S_fwd–vs–S_rev / S_fwd–vs–T)

| Metric | Ratio | 95% CI | Seeds with ratio < 1 |
|---|---|---|---|
| angle n=2 | **1.70** | [1.60, 1.84] | 0 / 30 |
| angle n=3 | **1.52** | [1.41, 1.64] | 0 / 30 |
| L2        | **1.70** | [1.54, 1.84] | 1 / 30 |

Every one of 30 seeds shows angle-based ratios well above 1. This is
**the opposite direction** of the paper's B6 result (0.86 at n=1).

## What this actually tests

The 1.70 ratio at n=30 is not evidence that "order dominates content."
It is evidence that under the current sampling design, the recency of
the LAST wake-trained task dominates the idle state.

### Why the sampling matters

The stream generator (`sample_specs` in `src/tasks.py`) partitions
[0.5, 20] Hz into K = 5 log-spaced bins and draws one freq per bin. This
produces a **deterministic ordering pattern**:

- `S_fwd` always ends at the top bin (last freq ≈ 10–20 Hz).
- `S_rev` always ends at the bottom bin (last freq ≈ 0.5–1 Hz).
- `T` also always ends at the top bin (same construction).

Per-seed check from the JSON (mean over 30 seeds):

- last freq of `S_fwd`: 13.1 Hz (range [9.65, 19.59])
- last freq of `S_rev`: 0.70 Hz (range [0.51, 1.03])
- last freq of `T`:    14.2 Hz (range [10.00, 19.99])

If the trained network's idle state is dominated by the most recent
task (recency bias), we would predict divergence to scale with the gap
between last-task frequencies:

- `S_fwd` vs `S_rev`: 13 Hz vs 0.7 Hz — huge gap → large angle
- `S_fwd` vs `T`:    13 Hz vs 14 Hz — small gap → small angle

Ratio > 1 falls out mechanically. The experiment as designed can't
distinguish "order matters (beyond recency)" from "recency matters
(and everything else is a residue)."

### The paper's B6 result is consistent with recency too

The paper used fixed streams:

- A = [1.0, 1.5, 2.0, 1.0, 2.0]      last: 2.0 Hz
- D = [2.0, 1.5, 1.0, 2.0, 1.0]      last: 1.0 Hz (reversed A)
- B = [8.0, 12.0, 6.0, 8.0, 6.0]     last: 6.0 Hz

Under recency dominance:

- angle(A, D): last-freq gap = |2 − 1| = 1 Hz → smallish angle
- angle(A, B): last-freq gap = |2 − 6| = 4 Hz → bigger angle
- Predicted ratio angle(A, D) / angle(A, B) < 1  ✓ paper's 0.86

Both the paper's B6 and my hardened v1 are consistent with a single
mechanism: the idle state after 5 wake-sleep cycles is dominated by
the most recent task. They land on opposite sides of ratio = 1 only
because the paper's last-freq gap was small for the reversed pair
(1 Hz apart) and large for the unrelated pair (4 Hz apart), while
mine is large for the reversed pair and small for the unrelated pair.

## What the paper can and cannot claim

**Confirmed at n=30:**

- Wake-sleep training over 5 cycles produces idle states whose
  divergence between two networks scales strongly with the difference
  between their most recent training tasks.
- All 30 seeds show angle(same-content-reversed) > angle(different-
  content-same-recency).

**Not confirmed (paper's original framing collapses):**

- "The ORDER of experience produces a signature independent of content."
  The current experiment cannot distinguish "order matters" from
  "the last cycle matters and everything else is decoration."
- "History accumulates into individuality." B3 already died. This
  experiment reinforces the accumulation story is not what's producing
  the divergence.

**New framing available (if `v2` below confirms it):**

> "Under wake-sleep training, the idle state is dominated by the most
> recent task. Divergence between networks is well predicted by the
> difference between their last training tasks; the effect of earlier
> history is subtle and requires a design that isolates it from
> recency."

## Next step: order_effect_hardened_v2

Redesign the streams to isolate ORDER from RECENCY:

- Per seed, sample K=5 freqs randomly.
- `S_perm_A` = random permutation of these K freqs.
- `S_perm_B` = a DIFFERENT random permutation of the SAME K freqs.
- `T_perm`   = a random permutation of a DIFFERENT K freqs sampled
  from the same distribution.

Then the last-freq of `S_perm_A`, `S_perm_B`, and `T_perm` are all
drawn from the same distribution — no systematic bias.

- angle(`S_perm_A`, `S_perm_B`): pure ORDER effect (same freqs,
  different order, random last-freq gap distribution).
- angle(`S_perm_A`, `T_perm`): CONTENT variation with matched
  distribution (different freqs, random last-freq gap).
- Ratio > 1 with tight CI → order matters more than content variation
  even after matching recency.
- Ratio ≈ 1 → the paper's B6 result was recency, order does not add
  anything measurable at L=5.

## Open questions

1. **Does the order effect appear at longer L?** At L=5 recency
   naturally dominates. At L=30, sleep-consolidated dynamics may
   integrate history further back and reveal an order effect above
   recency.
2. **Anchored-last-cycle test**: force `S` and `S_rev` to end at the
   same freq (reverse only the middle three cycles). If divergence
   drops to noise, recency was the whole story. If a substantial
   angle remains, earlier order matters too.

Both are Phase B extensions after `order_effect_hardened_v2` runs.

## Impact on the paper

- The paper's specific ordinal claim ("ratio 0.86, order matters less
  than content") is a byproduct of stream choice, not a robust finding
  about the mechanism. Drop the specific number.
- The general claim "history shapes idle state" survives only in the
  narrow form "the most recent trained task shapes idle state" until
  `v2` says otherwise.
- If `v2` shows ratio ≈ 1: the paper cannot claim ORDER as a
  first-class dimension of individuality separate from recency. That
  substantially weakens Path A (accumulation-of-individuality) and
  makes Path B (idiosyncratic-fingerprint) the honest framing.
