# Baseline richness claim correction

**Date**: 2026-05-26
**References**: `results/week22_2026/benchmark_20260526_v1.json`

## What changed

The original Stage 1 result reported LeakyRNN idle richness ~0.3 vs GRU richness ~10⁻⁶,
described as a "50,000× improvement". This number sat in the README, paper draft,
and architecture doc for several weeks.

Re-checking the consolidated benchmark JSON shows the GRU comparison was unfair:

- The 10⁻⁶ richness was an **untrained** GRU (or one where the feedback loop
  never escaped the near-zero fixed point during a brief Stage 1 probe).
- The 0.3 richness was the **trained** LeakyRNN at its best τ/SR config.

When both are trained on the same task (freq=3.0, 300 steps), the gap is:

- Trained GRU richness: 0.207
- Trained LeakyRNN, best (τ=2.0, SR=1.05): 0.324
- Ratio: 1.6×

## What this means

The architecture claim doesn't collapse, but its framing does. "GRUs cannot
sustain idle dynamics" was overstated. The accurate version: trained GRUs do
produce some idle activity, but the leaky integrator does it more naturally
(idle dynamics are present from initialisation, not just emergent from training).

The decoder-accuracy result (1.000 across 4 streams) is the load-bearing part
of the paper. It is unaffected by this correction.

## Action taken

- README, architecture.md, and paper.md updated to report 1.6× with both numbers
  shown. Untrained collapse mentioned as context, not as the headline.
- `run_baselines.py` added to do this properly: VanillaRNN, GRU, LSTM,
  FrozenESN, LeakyRNN compared on identical training and identical idle measurement.
- Once `run_baselines.py` runs, the new Figure 1 will replace the misleading
  "GRU baseline at zero" figure with a proper 5-way comparison.

## Open question

If FrozenESN matches LeakyRNN on idle richness and decoder accuracy, then the
contribution is much weaker — the limit-cycle attractor would be a property
of any random reservoir at SR≈1, not specifically of trained `W_rec`. We need
to see those numbers before the next paper revision.
