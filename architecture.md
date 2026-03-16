# Architecture Design Decisions

## Why a Leaky Integrator RNN, not a GRU

The first architecture we tried was a GRU with a self-feedback loop. It failed immediately: idle richness ~0.000004. The GRU's update gate (mean activation ~0.5) suppresses autonomous dynamics regardless of spectral radius ,  it's designed to forget slowly, not to oscillate.

The leaky integrator has no gating:

```
x(t+1) = (1 - 1/τ) · x(t) + (1/τ) · tanh(W_rec · x(t) + W_in · u(t) + b)
```

The leak rate `1/τ` directly controls the memory timescale. With τ=5.0, the network integrates over ~5 timesteps, producing rich autonomous oscillations (richness 0.15–0.32). This is the architecture Jaeger used in Echo State Networks and Maass used in Liquid State Machines.

**Config**: τ=5.0, SR_init=0.95, hidden_dim=128, input_dim=16, output_dim=16.

## Why train W_rec (not freeze it like a classical ESN)

Classical ESNs freeze the reservoir and only train the readout. We train W_rec because we want the network to develop task-specific attractor structures ,  not just use generic reservoir dynamics. Post-training spectral radius rises to 1.6–2.8 (supercritical), placing the network at the edge of chaos. This is a feature: it means the network has learned to sustain oscillations, not just respond to input.

## Self-feedback loop

Output y(t) = W_out · x(t) feeds back as input u(t) = y(t-1) during idle operation. This closes the loop: the network runs on its own output. The feedback connection adds a rank-one component to the effective connectivity (Mastrogiuseppe & Ostojic 2018), producing low-dimensional, interpretable dynamics.

During wake (external input present), the input is a blend: `α · x_ext + (1-α) · x_fb`. During idle (no external input), only the feedback path is active.

## Sleep phase design

**Rule**: ΔW_rec = η_eff · (H_c^T H_c) / T − η_eff · δ · W_rec

where H_c is the mean-centred hidden state trajectory over the sleep episode, T=600 steps, η_eff = η · ACh_gate.

Three design choices:

1. **Driven by recurrent dynamics, not stored data.** The network runs idle and the resulting limit cycle IS the replay signal. No stored input statistics (contrast: Tadros et al. 2022 use averaged input statistics from past tasks).

2. **Averaged over a full cycle, not per-step.** Stage 1A showed idle oscillations have period ~300 steps at τ=5. Averaging over 600 steps (2 cycles) captures the temporal structure of the attractor rather than a snapshot.

3. **ACh gate on learning rate.** η_eff = η · ACh_gate. During sleep, ACh_gate=0.3 (low acetylcholine → slow consolidation). During wake, ACh_gate=1.0 (high acetylcholine → full plasticity). This follows Hasselmo's cholinergic modulation of cortical plasticity (Avery & Krichmar 2017).

**Optimal config** (from Stage 3C parameter sweep, 16 configurations): η=0.01, decay=0.001.

## Divergence metric: subspace angle

Raw L2 distance between idle trajectories is a poor metric ,  it conflates magnitude with direction, and two networks can have very different trajectories while encoding the same thing. The primary metric is the **subspace angle** between the top-3 principal components of the two networks' idle trajectories:

```python
_, _, Va = np.linalg.svd(states_a - states_a.mean(0))
_, _, Vb = np.linalg.svd(states_b - states_b.mean(0))
sv = np.linalg.svd(Va[:3] @ Vb[:3].T, compute_uv=False)
angle = np.arccos(np.clip(sv, -1, 1)).mean() * 180 / np.pi
```

This measures how differently the two networks organise their variation ,  rotation-invariant and scale-invariant. L2 is reported as a secondary metric (measures positional divergence; subspace angle measures directional divergence ,  they dissociate under sleep, see paper Section 6.3).

## Known limitations

- Sleep does not reduce catastrophic forgetting in the task-performance sense (Stage 2, E3). The Hebbian update (~ΔW_norm 0.02) is small relative to gradient updates during wake training. Sleep consolidates dynamic identity (W_rec attractor structure), not task performance (W_out readout).
- The idle oscillation frequency is compressed ~300× relative to training frequency. This may be tunable via feedback gain but has not been characterised.
- All experiments use sinusoidal prediction as the wake task. Generalisation to other task types is untested.
