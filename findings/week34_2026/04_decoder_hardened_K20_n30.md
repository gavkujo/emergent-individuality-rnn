# Decoder at K=20, n=30 seeds: fingerprint is idiosyncratic, not shape-based

**Date**: 2026-08-23
**References**:
  - `results/week34_2026/decoder_hardened_20260823_v1.json`
  - `results/_checkpoints/decoder_hardened_v1/seed_0000.json` … `seed_0029.json`
  - Code: `repo/src/experiments/decoder_hardened/`

## The setup

Industry-standard replication of the paper's B2 idle-state decoder claim.

- **K = 20** sinusoid tasks, frequencies log-uniformly sampled from
  [0.5, 20] Hz (well-separated in log-space; seed 0).
- **n = 30** independent random inits (torch seeds 0..29).
- **hidden_dim = 256** (up from 128), otherwise canonical LeakyRNN
  config (τ=5, target_sr=0.95, α=0.8, W_in and W_out scale 0.1).
- **Wake-only training**: 300 BPTT steps at lr=3e-3, no sleep phase.
- **Idle trajectory**: 400 steps, last 300 used for decoding.
- **Chance level**: 1/K = 5%.
- Decoder heads: ridge (λ=1.0, closed-form), 5-fold CV. MLP (64 hidden,
  ReLU, full-batch SGD, 200 epochs).
- Statistics: bootstrap 95% CI over the 30 per-seed accuracies, 2000
  resamples.

## Results

### Within-seed decoding (per-seed metric averaged over 30 seeds)

| Metric | Mean | 95% CI | Per-seed range |
|---|---|---|---|
| ridge main         | **1.0000** | [1.0000, 1.0000] | all 30 seeds = 1.000 |
| ridge shuffled     | **1.0000** | [1.0000, 1.0000] | all 30 seeds = 1.000 |
| MLP main           | 0.3177 | [0.2987, 0.3371] | 0.233 – 0.434 |
| main − shuffled    | **0.0000** | [0.0000, 0.0000] | exactly 0 every seed |

### Cross-seed pooled controls (leave-one-seed-out)

| Control | Mean accuracy | Chance | Feature |
|---|---|---|---|
| weight-norm decoder    | **0.050** | 0.050 | ‖W_rec_final‖ (1-D) |
| mean-state decoder     | **0.045** | 0.050 | mean idle state (256-D) |

## Interpretation

### 1. Cross-network mapping is idiosyncratic, not universal

The **mean-state decoder** is the load-bearing new measurement. It takes
per-network scalars (one 256-D mean idle-state vector per (seed × task)
pair, one sample per class per seed), and asks: given how 29 networks
map task → mean-idle-state, can we predict the 30th network's mapping?

Answer: **no**, mean 4.5% (≈ chance for K=20). 30 folds, per-fold range
[0%, 15%]. The class centroids in idle-state space do not cluster by
task; instead they scramble seed-by-seed. Under one random init, task_k
lands in region R_k; under another init, task_k lands somewhere else
entirely.

Consequence: the fingerprint is **per-network task-specific** but not
**universally task-encoding**. Within a network trained on all 20 tasks
(one per network under a fixed init), each task's idle geometry is a
distinct blob in state space. Across networks with different inits,
those blobs are not aligned — task_k under seed 0 and task_k under seed 1
share no common region.

This is a **subtle but important** distinction from the framing implied
by a single-seed n=1 decoder = 1.000: that experiment says only "within
these specific networks, tasks are separable," not "task history has a
signature."

### 2. Temporal ordering carries no information

Ridge main and ridge shuffled are both exactly 1.000, across every seed.
Shuffling timesteps within each idle trajectory before pooling into the
decoder does not change accuracy. Since the ridge classifier operates on
individual timesteps, this means:

> The per-timestep activation distribution alone is linearly separable
> across the 20 classes; the temporal structure of the idle trajectory
> is not required.

Consequence: the paper's language around "trajectory geometry" or
"limit-cycle shape encodes history" is **not supported** by this
experiment. What is supported: per-timestep activation marginals differ
enough across trained networks that a linear classifier separates them.

This is consistent with each network becoming an autoregressive pattern
generator over-fit to its trained frequency; idle → generates that
frequency; per-timestep hidden state points lie on a distinct limit
cycle for each frequency. The classifier can pin down the limit cycle
from any single frame.

### 3. Confounds ruled out

Both scalar and vector controls of Vulnerability 2 come back clean:

- **Weight norm**: exactly 0.050 in every leave-one-seed-out fold. Even
  though wake training on different tasks changes ‖W_rec‖ by cycle,
  the norm carries no class information at K=20.
- **Mean activation**: at chance. Combined with the shuffled ≈ main
  result, this says the discriminative signal is not "one scalar per
  unit averaged over time" but something richer at the per-timestep
  level.

The decoder is not being fooled by a trivial confound. It's reading
actual geometry — just, that geometry is not shared across networks.

## What the paper can and cannot claim after this

**Supported claims:**
- Each wake-trained LeakyRNN develops a task-linked idle-state geometry
  that is linearly separable at K=20 with chance = 5%. This survives at
  n=30 seeds with a bootstrap 95% CI of [1.000, 1.000].
- The fingerprint is not attributable to weight-norm or mean-activation
  confounds.

**Not supported (needs to be dropped or reframed):**
- "The idle trajectory shape encodes history." Shuffling timesteps
  destroys shape and does not change decoder accuracy.
- "Task history has a universal signature in idle state." Across
  networks with different inits, the task → idle-state mapping does not
  transfer; mean-state pooled decoder is at chance.
- Anything reading the K=4 → 1.000 result as evidence of a canonical
  representation.

**New framing available:**
> "Under the wake-only training regime, each network becomes an
> autoregressive pattern generator specialised to its trained task. The
> idle-state signature is highly linearly decodable within the network
> (K=20, ridge = 1.000 across n=30 seeds) but is idiosyncratic across
> inits: the task-to-signature mapping does not generalise (leave-one-
> seed-out mean-state decoder at chance). Individuality here means
> 'each network has its own task-linked geometry,' not 'tasks have a
> canonical geometry across networks.'"

## Open questions surfaced

1. **Does sleep change this?** This experiment was wake-only. Sleep
   phase might produce trajectory-shape signals that shuffled-time
   would destroy. The next hardened experiment (`sleep_effect_hardened`)
   should re-run the same decoder with sleep, and compare the shuffled
   gap. If sleep creates trajectory structure, we'll see it there.
2. **Does history alignment help?** In the paper's original B2 stream
   variant, each network is trained on a SEQUENCE of tasks (not one).
   Does pooling across seeds recover the fingerprint when the pooling
   is over full histories rather than single tasks? Needs a
   `decoder_stream_hardened` experiment.
3. **Non-linear decodability**: MLP is at 0.318, below ridge. This is
   an MLP-training issue (full-batch SGD at lr=0.01 doesn't converge in
   200 epochs at 6000 samples in 256-D); the MLP is not a scientific
   result. Rerun with a stronger head (Adam, lr scheduling, or more
   epochs) as a code fix before drawing conclusions from it.

## Compute + hardware

- Device: MPS (Apple Silicon).
- Total wall-clock: 22,667 s ≈ 6.3 hours for 30 seeds.
- Per-seed wall-clock: seeds 0-9 and 19-29 landed around 500 s each;
  seeds 10-18 hit 887-1656 s, likely due to MPS memory not being
  reclaimed across models. Recovers on its own by seed 19. This
  reinforces the case for running on CUDA where the memory allocator
  is better-behaved.

## What changes in the todo_hardening checklist

- Phase B item **decoder_hardened**: mark done. Result: within-seed
  claim survives, cross-network claim requires reframing.
- Phase B item **sleep_effect_hardened**: elevated priority. Its
  purpose is now not just "does sleep amplify divergence" but "does
  sleep create the temporal-structure signal that wake-only lacks."
- Phase F paper rewrite: the "trajectory geometry" language must be
  replaced with something honest to the shuffled-control result.
