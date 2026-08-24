# Sleep-effect on divergence at n=30: inconclusive, effect indistinguishable from noise

**Date**: 2026-08-24
**References**:
  - `results/week35_2026/sleep_effect_hardened_20260824_v1.json`
  - Code: `repo/src/experiments/sleep_effect_hardened/`

## The setup

Industry-standard replication of the paper's B5 sleep-effect claim
("adding a Hebbian sleep phase between wake-training cycles amplifies
the divergence between networks trained on different streams").

- **n = 30** independent random inits.
- **Stream length L = 5** wake-sleep cycles.
- Per-seed streams: two independent per-seed sinusoid streams `S_A`,
  `S_B` (freqs log-uniform on [0.5, 20], one per log-bin).
- Four networks per seed, all from the SAME
  `torch.manual_seed(seed)` init:
  - `w_A`: wake+sleep on `S_A`
  - `w_B`: wake+sleep on `S_B`
  - `n_A`: wake-only on `S_A`
  - `n_B`: wake-only on `S_B`
- Metrics: subspace angle at n=2 and n=3, L2 divergence.
  Paired delta = with_sleep − without_sleep per seed. Bootstrap 95%
  CI over 30 per-seed paired deltas, 2000 resamples.

## Results

| Metric | with_sleep | without_sleep | paired Δ (with − without) | 95% CI | Seeds where Δ > 0 |
|---|---|---|---|---|---|
| angle n=2 | 45.2° | 44.0° | **+1.16°** | [−0.36, +2.91] | 14 / 30 |
| angle n=3 | 52.9° | 51.0° | **+1.90°** | [−0.43, +4.08] | 19 / 30 |
| L2        |  4.62 |  4.25 | **+0.37**  | [−0.14, +0.88] | 20 / 30 |

Every CI crosses zero. Every point estimate is positive but small. The
sign test on 30 seeds is close to 50/50 for angle_n2 (14/30) and only
mildly positive for the other two (19-20/30).

## Verdict

At n=30 seeds, the sleep phase has **no measurable effect on the
magnitude of idle-state divergence** between networks trained on
independent streams. All three metrics are inconclusive (CI crosses 0).

The paper's B5 point-estimate claim ("sleep amplifies divergence")
does not survive multi-seed replication. What was reported at n=1 is
within the seed-to-seed noise floor of a 30-seed sample from the same
protocol.

## Interpretation

This does NOT mean the sleep phase does nothing. It means the sleep
phase's contribution to **cross-network divergence at L=5 cycles** is
not distinguishable from noise. The sleep phase may still be doing
important work:

- **Consolidating the limit cycle**: sleep drives the network's
  autonomous idle state toward a stable attractor. This experiment
  measures the DIFFERENCE between two networks, not the QUALITY of
  each network's idle state. A separate experiment (sleep vs
  no-sleep richness/attractor quality) would test that claim.
- **Working at longer L**: at L=5, recency dominates (see
  `05_order_effect_hardened_v1.md`). The sleep phase's cumulative
  contribution may only emerge at longer streams where the Hebbian
  update integrates over more cycles.
- **Working differently on different metrics**: this experiment
  measures divergence between INDEPENDENT-stream pairs. Sleep may
  amplify individuality more sharply on quantities like decoder
  accuracy, order effects, or downstream learning behaviour
  (Checkpoint 1) than on subspace angle.

## What the paper can and cannot claim

**Not supported (drop from paper):**

- "Sleep phase amplifies L2 divergence by ~10%" (n=1). At n=30 the
  paired delta CI covers zero.
- Any wording that treats sleep-amplification-of-divergence as an
  established finding.

**Neutral framing that survives:**

- Sleep is a design ingredient that consolidates the limit cycle
  after wake training. Its effect on cross-network divergence at
  L=5 wake-sleep cycles is not statistically distinguishable from
  zero.
- The role of sleep in shaping longer-term individuality is an
  open question, best addressed by (a) longer streams and (b)
  downstream-task tests (Checkpoint 1).

## Related caveat from finding 05

At L=5, recency dominates idle-state geometry (finding
`05_order_effect_hardened_v1.md`). Since recency effects are much
larger than any accumulated cross-cycle contribution, the sleep
phase's cumulative effect is easily lost in the last-cycle noise.
The natural follow-up is either:

- (a) `sleep_effect_hardened_v2` at L = 20 or 30 cycles — larger
  window for accumulated sleep contribution to show up.
- (b) A different metric altogether: does sleep improve the
  learnability of the fingerprint (Checkpoint 1's central claim)?

## Impact on the paper

The paper's B5 section, as currently written, cannot be defended at
n=30. Options for Phase F rewrite:

1. **Drop B5 entirely** and treat sleep as an unquantified design
   choice.
2. **Reframe as ablation**: report the paired-delta CI honestly and
   say sleep does not amplify divergence at L=5.
3. **Move sleep to a subsection about mechanism** (limit-cycle
   consolidation) rather than a subsection about individuality
   magnitude.

Option 2 is the honest position. Option 3 opens the door to the
Checkpoint 1 story where sleep may have a downstream, functional
role even if it does not amplify divergence per se.
