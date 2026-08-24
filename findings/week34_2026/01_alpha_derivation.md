# α derivation: the theorem doesn't apply to the current codebase

**Date**: 2026-08-22
**References**:
  - `results/week34_2026/alpha_init_20260822_v1.json`
  - `supporting/working/alpha_derivation_proof_yours.md` (Parts 3-8)
  - `figures/alpha_init/` (four PNGs)
  - Code: `repo/src/alpha.py`, `repo/src/experiments/alpha_init/`

## The claim

For a leaky-integrator RNN with self-feedback,
        `A(α) = (1 − 1/τ)·I + (1/τ)·[W_recᵀ + (1 − α)·W_outᵀ·W_inᵀ]`
there is a unique α* ∈ (0, 1) with ρ(A(α*)) = 1 provided ρ(W_rec) < 1 (H1)
and ρ(A(0)) > 1 (H2). Proof in worksheet §4. This finding reports what
happens when we try to apply the theorem to the actual weights the codebase
uses.

## What the data shows

Configuration: `LeakyRNN(input_dim=16, hidden_dim=128, output_dim=16,
tau=5.0, target_sr=0.95)`, torch seed 42, W_in and W_out sampled with entry
scale 0.1 (the current codebase default).

### At the legacy scale (s = 0.1)

| Quantity | Value | Hypothesis |
|---|---|---|
| ρ(W_rec)                        | 0.950   | (H1) ✓ |
| ρ(A(0))                         | 0.994   | (H2) ✗ |
| ρ(A(0.8))  ← legacy operating α | 0.981   | — |

The linearised operator ρ(A(α)) never reaches 1 for any α ∈ [0, 1]. The
sweep is even non-monotone, dipping to a minimum ≈ 0.972 near α ≈ 0.6.
**No α* exists in (0, 1) at the codebase's current W_in / W_out scale.**

### After joint rescaling W_in, W_out by a common factor s

Grid + bisection over s ∈ [0.05, 2.0]:

| Transition | s |
|---|---|
| smallest s satisfying (H2) alone | ≈ 0.117 |
| smallest s satisfying empirical monotonicity | ≈ 0.484 |
| **s* = smallest s satisfying both**            | **0.4516** |

At s = s* = 0.4516:

| Quantity | Value |
|---|---|
| ρ(W_rec)                                              | 0.950 |
| ρ(A(0))                                               | 2.279 |
| **α***                                                | **0.9359** |
| **ρ(A(α*))**                                          | **1.00000000** (`|·| − 1| = 3.5e-9`) |
| bisection iterations                                  | 25 |
| (E.ii) min inner product over sweep                   | 0.507 (> 0) |
| empirical monotonicity worst gap (ρ_{k+1} − ρ_k)      | −0.013 (all ≤ 0) |

The theorem's uniqueness argument (worksheet §4.4) is satisfied cleanly at
this scale.

### Post-training drift (300 steps of wake_phase on freq=3 sine)

Both configs are trained identically; only the init differs.

| Metric | Legacy (s=0.1, α=0.8) | Principled (s=0.45, α=0.94) |
|---|---|---|
| ρ(W_rec) init                                | 0.950 | 0.950 |
| ρ(W_rec) after 300 steps                     | 1.796 | 3.397 |
| ρ(A(operating α)) init                        | 0.981 | 1.000 |
| ρ(A(operating α)) after 300 steps             | 0.963 | 1.250 |
| ‖ΔW_rec‖_F                                    | 0.66  | 4.15  |
| ‖ΔW_out‖_F                                    | 0.29  | 3.66  |
| theorem still applies post-training?          | no ((H1) fails) | no ((H1) fails) |

The legacy config drifts *away* from the edge (ρ went from 0.98 to 0.96,
i.e., more subcritical). The principled config drifts *past* the edge
(ρ went from 1.00 to 1.25, super-critical).

### Linearisation validity (idle, 300 steps after training)

| Quantity | Legacy | Principled |
|---|---|---|
| ‖h‖ mean          | 3.39  | 8.61  |
| ‖h‖ max           | 3.61  | 9.36  |
| tanh'(z) mean     | 0.891 | 0.209 |

In the principled config, the average unit's `tanh'(z)` is 0.21, i.e., the
tanh nonlinearity is strongly compressing the effective gain. The effective
operator has spectral radius roughly `tanh'(z) · ρ(A(α)) ≈ 0.21 · 1.25 ≈
0.26` — well below the linearised prediction. Saturation acts as a nonlinear
regulator.

The legacy config sits in a milder regime (`tanh'(z) ≈ 0.89`); the
linearised analysis is reasonably close to the effective dynamics there.

## What this means

Three things.

### 1. The current codebase is not "at edge of chaos" at init

Whatever else the paper claims about the mechanism, it should not claim
edge-of-chaos initialisation. The linearised operator ρ(A(0.8)) = 0.981 at
seed=42, with no α satisfying ρ(A(α)) = 1 anywhere in [0, 1]. The choice
of α = 0.8 was arbitrary and did not come from the theorem.

This is directly relevant to `todo_hardening.md`'s critical item ⚠ *"Edge
of chaos claim was verified only for τ=5"*. Actually the claim wasn't
verified there either. It's simply not a property of the current init.

### 2. The theorem tells us how to get to edge of chaos: rescale

To make the α theorem apply at init, W_in and W_out both need to be
scaled up by ≈ 4.5×. At (s=0.4516, α=0.9359) the theorem cleanly holds,
with ρ(A(α*)) = 1 verified to 8 decimal places. This gives the paper a
principled `(W_in scale, W_out scale, α)` triple to compare against the
legacy `(0.1, 0.1, 0.8)` triple.

Whether this rescaling meaningfully affects the downstream individuality
metrics is *not* what this experiment measured. That's the workshop-scope
sanity sweep — todo item 9.

### 3. The linearised theorem is only about initialisation

Post-training, ρ(W_rec) grows past 1 in both configs, so (H1) fails on
the trained weights and `compute_alpha` raises. And tanh saturation
reshapes the effective dynamics — heavily so in the principled config.
The α theorem is honestly a *design tool for initialisation*, not a
description of steady-state operation. Any post-training "edge of chaos"
argument requires additional machinery (adaptive α, nonlinear analysis,
or an argument that init-time criticality biases training toward a
useful steady state).

## What changes because of this

### In the paper (workshop draft)

- Remove all "edge of chaos" claims that aren't hedged as init-time
  properties.
- Report the actual ρ(A(0.8)) = 0.981 at init if it appears anywhere.
- If the theorem is included, state its scope clearly: init-time
  linearised operator, requires (H2) which needs rescaling.

### In the codebase

- `src/alpha.py` is now the library for the theorem.
- `src/experiments/alpha_init/` is now the experiment that instantiates it.
- LeakyRNN / VanillaRNN / FrozenESN constructors are unchanged — I did
  not switch the default to (s*, α*) because that would break every
  existing benchmark JSON and require re-running everything. That switch
  is deferred to whichever experiment first uses the principled config
  end-to-end (probably the workshop-scope 3-point sanity sweep, todo #9).

### In the FYP scope

- The `W_in / W_out scale` items in `todo_hardening.md` (High section)
  are now partially answered: joint scale s* = 0.4516 satisfies the α
  theorem cleanly. Whether that's the "right" scale for downstream
  metrics is the sanity sweep's job.

## Caveats

- Single seed (42). The scale s* and value α* depend on the specific
  W_rec eigenstructure at that seed. Multi-seed replication is on the
  todo list. A stable α* range across seeds is what the workshop paper
  would need to cite as "the principled α" rather than a single value.
- The (E.i) "eigenvalue gap" check in the JSON reports 0.0 across the
  sweep. This is a numerical artefact: random real W_rec has a
  dominant complex-conjugate pair with equal magnitudes by
  construction. The gap-between-top-two-magnitudes is zero for a pair;
  the check that actually matters (E.ii inner product > 0 and empirical
  monotonicity) holds cleanly. The current code's E.i reporting is
  misleading; the JSON should be read for the E.ii + monotonicity
  numbers.
- Post-training drift is measured on a single training task (freq=3
  sinusoid, 300 steps). Longer training or different tasks may drift
  further or differently.
- The "linearisation validity" measurement assumes idle operation post-
  training. During wake operation with `x` present, ‖h‖ may differ.

## Open question

Does the choice between (s=0.1, α=0.8) and (s=0.45, α=0.94) meaningfully
change the individuality metrics (decoder accuracy, subspace angle
accumulation, order effect)? Answering this is the workshop-scope
sanity sweep (todo item 9). If the metrics are similar, the workshop
paper can present the principled config as a cleaner drop-in replacement.
If they differ, the choice becomes a scientifically-loaded design
decision and the paper needs to argue for one or the other explicitly.
