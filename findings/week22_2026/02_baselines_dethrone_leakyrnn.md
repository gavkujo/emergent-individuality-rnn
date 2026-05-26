# Baselines reframe what the architecture claim actually is

**Date**: 2026-05-26
**References**: `results/week22_2026/baseline_comparison_20260526_v1.json`

## What the data shows

Five architectures, identical self-feedback loop, identical training (300 steps,
freq=3.0 sine), identical idle measurement (300 idle steps).

### Idle richness (std of hidden state over 300 idle steps)

| Architecture | Untrained | Trained |
|--------------|-----------|---------|
| LeakyRNN     | **0.209** | 0.324   |
| VanillaRNN   | 0.169     | 0.297   |
| FrozenESN    | 0.016     | 0.262   |
| GatedRNN (GRU) | 0.004   | 0.206   |
| LSTM         | 0.003     | **0.333** |

### Decoder accuracy on idle states (4 frequencies, chance=0.250)

All five architectures: **1.000.**

## What this means

Two claims that were in the paper need to change.

### Claim 1: "LeakyRNN produces uniquely rich idle dynamics" — FALSE

After training, LSTM (0.333) and LeakyRNN (0.324) are tied within noise.
VanillaRNN (0.297) is close behind. Even FrozenESN (0.262), where W_rec
never changes, reaches 0.26.

The trained-richness ordering does not support an architecture-specific claim.
The four trained architectures all sustain similar levels of idle activity.

What LeakyRNN *is* uniquely good at is **untrained** richness (0.209 from
initialisation, vs ≤0.17 for everything else). This is a real but narrower
claim: leaky integrators have idle dynamics by default; gated architectures
have to be trained into them.

### Claim 2: "Decoder accuracy is the signature of our architecture" — FALSE

All five architectures hit 1.000 on the 4-frequency decoder task. The
decodability of experiential history is **not** a property of LeakyRNN
specifically. It's a property of:

  (a) the self-feedback loop (shared by all five),
  (b) trained W_out being task-specific (shared by all five),
  (c) the task being easy (4 well-separated frequencies, 128-dim state).

This last point matters. With four frequencies in a 128-dim hidden space,
any reasonable RNN with a task-specific readout will produce linearly
separable idle states. The benchmark doesn't discriminate between architectures.

## What the contribution actually is

We have a self-feedback architecture and a sleep mechanism that produces:

- accumulating divergence over wake-sleep cycles
- path-dependence (order matters)
- positional/directional sleep dissociation

These results have not been replicated across baselines yet. They are about
the wake-sleep cycle and the self-feedback loop, not specifically about leaky
integrators. **The contribution is the mechanism, not the cell type.**

For the paper this is a clean story: any rich enough RNN with a self-feedback
loop and our wake-sleep procedure will exhibit these properties. We use
LeakyRNN because it has good untrained dynamics (less dependence on training
to bootstrap the idle behaviour), but LSTM or VanillaRNN would likely produce
similar Stage 3 results.

## What we need to do

1. **Run Stages 2 and 3 across all five architectures.** The accumulation,
   path-dependence, sleep effect, and order effect benchmarks should all be
   measured per-architecture. If they hold across the board, the mechanism
   claim is strong. If they only hold for LeakyRNN, the architecture claim
   becomes load-bearing again and we keep the current framing.

2. **Make the decoder benchmark harder.** 4-class, 128-dim, 1.000 across the
   board is uninformative. Options:
   - More streams (16 or 32 instead of 4)
   - Smaller hidden_dim (e.g. 32 instead of 128)
   - Tighter frequency spacing (1.0, 1.1, 1.2, ... vs 1, 2, 4, 8)
   - Adversarial: streams that share most of their tasks, differ in one
   The goal is to find a regime where some architectures fail and others
   succeed, so the metric distinguishes them.

3. **Update Figure 1, README, paper Section 3.1, architecture.md.** The
   "GRU is bad" framing has to go. The honest framing is: leaky integrators
   are the only architecture with idle dynamics from initialisation, but
   after training, all five architectures sustain comparable richness; the
   contribution is the wake-sleep mechanism, not the cell.

## Caveats

- n=1 per architecture. Should run 5+ seeds before publishing.
- Single training task (freq=3.0). The "richness after training" number depends
  on the specific training task. A different task could re-order the architectures.
- Decoder accuracy is at ceiling everywhere — the metric is saturated. The
  fact that they all tie at 1.000 doesn't mean they're equally good at
  decodability, just that this particular benchmark can't tell them apart.
- FrozenESN at 1.000 is the most interesting datapoint and deserves its own
  follow-up: how is W_out alone enough to produce task-specific limit cycles?
  My best guess is that the self-feedback loop turns the trained W_out into
  an effective rank-(output_dim) modification of the recurrent dynamics, so
  changing W_out across tasks changes the effective dynamics matrix.

## Open question

The most surprising line in the data is FrozenESN reaching 0.262 trained richness
and 1.000 decoder accuracy with W_rec never updated. That's a much stronger
form of the original ESN claim: a fixed random reservoir is sufficient for our
entire individuality result, as long as the readout is trained. If this holds
under harder benchmarks, the paper's contribution simplifies to:
**"a trained readout, fed back through a fixed random reservoir, produces
experientially-divergent idle dynamics."** That would be a cleaner and stronger
claim than what we have now.
