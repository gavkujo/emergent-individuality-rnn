"""
Task family generators.

Every task family shares one interface:

    make_task(spec, T, batch, input_dim, seed, device) -> (inputs, targets)

`inputs`/`targets` are torch tensors of shape (T-1, batch, input_dim). The
target is the next-timestep signal (autoregressive prediction).

TaskSpec is a small dataclass identifying the family and its parameters.
Streams are just `list[TaskSpec]`.

Design notes:

  - Every signal is generated at unit-variance to keep decoder/richness
    numbers directly comparable across families.
  - Every family populates the full `input_dim`-channel signal from
    `input_dim` independent per-channel realisations of the same
    distribution (per-channel phase for sine/chirp, per-channel initial
    condition for pulse/Lorenz). This preserves an information-rich
    input where a scalar signal would give an effectively 1-D input.
  - `sample_spec(family, seed)` draws from a family-specific
    distribution suitable for decoder experiments (K distinct tasks
    per class, well-separated within reason).

Families:
    sine    — smooth periodic, frequency Hz
    chirp   — linearly-swept frequency, f_start → f_end
    pulse   — sparse binary pulse train, rate Hz
    lorenz  — chaotic 3D attractor projection, parameter ρ
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch


# Fundamental time unit. `dt = 0.01` gives ~100 samples per Hz, matching
# the resolution the pilot sine tasks were tuned at.
DT = 0.01


# ── TaskSpec ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskSpec:
    """Identifies one task instance.

    Attributes:
        family: one of {"sine", "chirp", "pulse", "lorenz"}
        params: family-specific parameters (all JSON-serialisable).
        label:  human-readable identifier (used in filenames, plots).
    """
    family: str
    params: dict
    label: str = ""

    def to_dict(self) -> dict:
        return {"family": self.family, "params": dict(self.params),
                "label": self.label}

    @staticmethod
    def from_dict(d: dict) -> "TaskSpec":
        return TaskSpec(family=d["family"], params=dict(d["params"]),
                        label=d.get("label", ""))


# ── Per-family generators ────────────────────────────────────────────────────
#
# Each generator returns an (T, input_dim) numpy array. The `make_task`
# wrapper below handles the T-1 target shift, batch broadcast, and torch
# conversion.

def _sine(params: dict, T: int, input_dim: int, rng: np.random.Generator) -> np.ndarray:
    """input_dim phase-shifted sines at one frequency, unit-variance."""
    freq = float(params["freq"])
    t = np.arange(T) * DT
    phases = rng.uniform(0, 2 * np.pi, size=input_dim)
    signal = np.sin(2 * np.pi * freq * t[:, None] + phases[None, :])
    # sine of unit amplitude has variance 1/2; scale to unit variance
    return signal * np.sqrt(2.0)


def _chirp(params: dict, T: int, input_dim: int, rng: np.random.Generator) -> np.ndarray:
    """input_dim linearly-swept chirps sharing (f_start, f_end), unit-variance."""
    f0 = float(params["f_start"])
    f1 = float(params["f_end"])
    t = np.arange(T) * DT
    # linear-chirp instantaneous phase: 2π * (f0·t + 0.5·(f1-f0)/T_total · t²)
    T_total = T * DT
    phases_offset = rng.uniform(0, 2 * np.pi, size=input_dim)
    inst_phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T_total * t ** 2)
    signal = np.sin(inst_phase[:, None] + phases_offset[None, :])
    return signal * np.sqrt(2.0)


def _pulse(params: dict, T: int, input_dim: int, rng: np.random.Generator) -> np.ndarray:
    """input_dim independent binary pulse trains at target rate `rate` Hz.

    Pulses are half-cosine-shaped windows of width `width_ms` ms to keep the
    signal differentiable. Zero-mean, unit-variance.
    """
    rate = float(params["rate"])
    width_ms = float(params.get("width_ms", 40.0))
    width_samples = max(int(round(width_ms / (1000.0 * DT))), 1)
    T_total = T * DT
    expected_pulses = int(round(rate * T_total))

    signal = np.zeros((T, input_dim), dtype=np.float64)
    # Half-cosine bump on [-1, 1]: 0.5·(1 + cos(π·x))
    xw = np.linspace(-1.0, 1.0, width_samples, endpoint=False)
    kernel = 0.5 * (1.0 + np.cos(np.pi * xw))

    for c in range(input_dim):
        # Sample pulse starts uniformly.
        n_pulses = max(rng.poisson(expected_pulses), 1)
        starts = rng.integers(0, max(T - width_samples, 1), size=n_pulses)
        for s in starts:
            signal[s:s + width_samples, c] += kernel

    # Normalize to zero-mean, unit-variance per channel
    signal = signal - signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    signal = signal / np.maximum(std, 1e-8)
    return signal


def _lorenz(params: dict, T: int, input_dim: int, rng: np.random.Generator) -> np.ndarray:
    """input_dim Lorenz-attractor trajectories at parameter ρ.

    Each channel is a different Lorenz simulation with random initial
    condition. We use only the x-coordinate to keep it a scalar per
    channel. Integrator: RK4 at step size DT / oversample.
    """
    rho = float(params["rho"])
    sigma = float(params.get("sigma", 10.0))
    beta = float(params.get("beta", 8.0 / 3.0))
    oversample = int(params.get("oversample", 4))
    h = DT / oversample
    burn_in = int(params.get("burn_in", 500))

    def rhs(state):
        x, y, z = state
        return np.array([sigma * (y - x),
                         x * (rho - z) - y,
                         x * y - beta * z])

    def rk4(state, h):
        k1 = rhs(state)
        k2 = rhs(state + h / 2 * k1)
        k3 = rhs(state + h / 2 * k2)
        k4 = rhs(state + h * k3)
        return state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    trajectories = np.empty((T, input_dim), dtype=np.float64)
    for c in range(input_dim):
        state = rng.normal(size=3) * 1.0
        state[2] += rho  # start near the attractor
        # Burn in to land on the attractor.
        for _ in range(burn_in):
            state = rk4(state, h)
        for t in range(T):
            for _ in range(oversample):
                state = rk4(state, h)
            trajectories[t, c] = state[0]

    # Normalize per channel to unit variance
    trajectories = trajectories - trajectories.mean(axis=0, keepdims=True)
    std = trajectories.std(axis=0, keepdims=True)
    trajectories = trajectories / np.maximum(std, 1e-8)
    return trajectories


_GENERATORS = {
    "sine":   _sine,
    "chirp":  _chirp,
    "pulse":  _pulse,
    "lorenz": _lorenz,
}


# ── Public interface ─────────────────────────────────────────────────────────

def make_task(spec: TaskSpec, T: int = 200, batch: int = 4,
              input_dim: int = 16, seed: int = 0, device=None
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (inputs, targets) tensors for one task instance.

    Args:
        spec:      TaskSpec identifying family + params.
        T:         number of raw timesteps to generate (target has T-1).
        batch:     batch dimension (signal is broadcast; each batch is
                   identical). Kept for interface compatibility with the
                   pilot make_sine_task.
        input_dim: number of independent channels.
        seed:      RNG seed. Determines per-channel phase/initial condition.
        device:    torch device or None.

    Returns:
        inputs:  (T-1, batch, input_dim) tensor.
        targets: (T-1, batch, input_dim) tensor (inputs shifted by +1).
    """
    if spec.family not in _GENERATORS:
        raise ValueError(f"unknown task family: {spec.family!r}")
    rng = np.random.default_rng(seed)
    signal = _GENERATORS[spec.family](spec.params, T, input_dim, rng)
    signal_t = torch.from_numpy(signal.astype(np.float32))
    signal_t = signal_t.unsqueeze(1).expand(T, batch, input_dim).contiguous()
    if device is not None:
        signal_t = signal_t.to(device)
    return signal_t[:-1], signal_t[1:]


# ── Distributions for sampling K task specs ─────────────────────────────────

def sample_specs(family: str, K: int, seed: int = 0) -> list[TaskSpec]:
    """Sample K task specs from the family's distribution.

    Distributions (chosen to be well-separated for a K-way decoder while
    staying in a physically reasonable band):
        sine   — freq ~ log-uniform on [0.5, 20] Hz
        chirp  — (f_start, f_end) with a 3–5x sweep ratio
        pulse  — rate ~ log-uniform on [0.5, 10] Hz
        lorenz — ρ ~ uniform on [15, 40] (chaotic band; classic value 28)

    Values within a family are sampled without replacement in log-space
    to guarantee separation.
    """
    rng = np.random.default_rng(seed)

    if family == "sine":
        # log-uniform separated in log-space to avoid near-duplicates
        edges = np.linspace(np.log(0.5), np.log(20.0), K + 1)
        freqs = np.exp(rng.uniform(edges[:-1], edges[1:]))
        return [TaskSpec("sine", {"freq": float(f)}, f"sine_{f:.2f}Hz")
                for f in freqs]

    if family == "chirp":
        edges_start = np.linspace(np.log(0.5), np.log(5.0), K + 1)
        edges_end = np.linspace(np.log(5.0), np.log(20.0), K + 1)
        f0s = np.exp(rng.uniform(edges_start[:-1], edges_start[1:]))
        f1s = np.exp(rng.uniform(edges_end[:-1], edges_end[1:]))
        return [TaskSpec("chirp",
                         {"f_start": float(a), "f_end": float(b)},
                         f"chirp_{a:.2f}to{b:.2f}Hz")
                for a, b in zip(f0s, f1s)]

    if family == "pulse":
        edges = np.linspace(np.log(0.5), np.log(10.0), K + 1)
        rates = np.exp(rng.uniform(edges[:-1], edges[1:]))
        return [TaskSpec("pulse", {"rate": float(r)}, f"pulse_{r:.2f}Hz")
                for r in rates]

    if family == "lorenz":
        edges = np.linspace(15.0, 40.0, K + 1)
        rhos = rng.uniform(edges[:-1], edges[1:])
        return [TaskSpec("lorenz", {"rho": float(r)}, f"lorenz_rho{r:.1f}")
                for r in rhos]

    raise ValueError(f"unknown family: {family!r}")


# ── Backward-compat shim ────────────────────────────────────────────────────
#
# Old code in `src.experiments.benchmark` etc. calls
# `train.make_sine_task(freq, ...)`. The old API is kept in `train.py`;
# this module simply provides a helper for callers who want the new API.

def sine_spec(freq: float) -> TaskSpec:
    """Convenience for legacy call sites."""
    return TaskSpec("sine", {"freq": float(freq)}, f"sine_{freq:.2f}Hz")
