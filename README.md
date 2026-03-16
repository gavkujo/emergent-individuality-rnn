# Emergent Individuality in Neural Networks via Persistent Internal State Dynamics

> **Draft (under review)**

Two recurrent networks, initialised identically, exposed to different sequences of experiences, develop measurably different internal dynamics. No explicit individuality objective. No different architectures. Just history.

---

## The core result

A linear decoder trained on idle hidden states (no input, network running on its own dynamics) identifies which of 4 distinct experiential streams a network underwent, with **1.000 accuracy** (chance: 0.250).

The same tasks in reversed order produce **86% of the divergence** of completely different tasks. The network encodes sequence, not just content.

---

## How it works

**Architecture**: Leaky integrator RNN with a self-feedback loop. Output feeds back as input during idle operation, sustaining autonomous limit cycle oscillations. The character of those oscillations encodes experiential history.

**Wake phase**: standard BPTT on a sinusoidal prediction task. The network learns; W_rec and W_out are updated.

**Sleep phase**: no external input. The network runs on its own recurrent dynamics for 600 steps. Local Hebbian consolidation is applied to W_rec, strengthening co-active connections, applying mild weight decay. No stored data, no backprop. Driven entirely by the network's own limit cycle.

The sleep phase is modelled on [Tadros et al. 2022](https://www.nature.com/articles/s41467-022-35266-6) and the Synaptic Homeostasis Hypothesis [Tononi & Cirelli 2014], with the key extension that consolidation is driven by recurrent dynamics rather than stored input statistics.

---

## Key results

| Benchmark | Result |
|-----------|--------|
| Idle richness: LeakyRNN vs GRU | 0.32 vs 0.000004 (~50,000× improvement) |
| Decoder accuracy (4 streams, 5 cycles each) | 1.000 (chance 0.250) |
| Divergence accumulation over 5 cycles | 54.47° to 62.17° subspace angle |
| Path-dependence (length 1 to 10) | 59.89° to 60.70° (growing, not plateauing) |
| Order effect (same tasks reversed) | 55.27° = 86% of different-task divergence |
| Sleep effect on L2 divergence | +10% vs no-sleep baseline |

---

## Repo structure

```
src/
  model.py       : LeakyRNN with self-feedback
  train.py       : wake/sleep training loop + utilities
  experiments/
    benchmark.py : reproduces all results in results/benchmark_results.json
    figures.py   : generates all figures from benchmark_results.json
figures/         : all plots (generated)
results/         : benchmark_results.json
paper.md         : full draft paper with figure placements
architecture.md  : design decisions and justifications
requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

# run all benchmarks (~20 min on CPU, ~5 min on MPS/CUDA)
python src/experiments/benchmark.py

# generate figures
python src/experiments/figures.py

# figures saved to figures/, results to results/benchmark_results.json
```

---

## Status

- [x] Stage 1: Baseline architecture (LeakyRNN, attractor characterisation)
- [x] Stage 2: Sleep/wake cycle (Hebbian consolidation, SR safety)
- [x] Stage 3: Experiential divergence (core novelty experiment)
- [ ] Stage 4: Neuromodulatory gates (DA/5HT/ACh/NE). **(in progress)**

---

## Citation

```
@misc{emergent-individuality-2026,
  title  = {Emergent Individuality in Neural Networks via Persistent Internal State Dynamics},
  year   = {2026},
  note   = {Draft under review}
}
```
