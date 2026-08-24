"""
Wake/sleep training loop and shared utilities.

Public functions:
    make_sine_task(freq, ...)                — legacy sinusoid task generator
    wake_phase(model, inputs, targets, ...)  — BPTT wake training
    sleep_phase(model, ...)                  — Hebbian sleep consolidation
    run_idle(model, steps, device)           — collect idle-state trajectory
    train_stream(...)                        — canonical wake/sleep loop
                                               over a `list[TaskSpec]`

Legacy metric re-exports (moved to `src.metrics`):
    subspace_angle, effective_dim, decoder_accuracy

`train_stream` is the entry point used by every hardened experiment. It
takes a task-family-agnostic `stream: list[TaskSpec]` (see `src.tasks`)
and produces per-cycle idle trajectories, ready for the divergence
metrics.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from src.tasks import TaskSpec, make_task
# Re-export metrics for backward compat with older experiments (benchmark,
# subspace_dim, accumulation_multiseed, sleep_order_multiseed) that import
# these names from `src.train`.
from src.metrics import subspace_angle, effective_dim, decoder_accuracy  # noqa: F401


# ── Legacy task generator ────────────────────────────────────────────────────

def make_sine_task(freq: float, T: int = 200, batch: int = 4,
                   input_dim: int = 16, seed: int = 0, device=None):
    """Sinusoidal prediction task. Kept for legacy experiment code.

    New experiments should use `src.tasks.make_task(spec, ...)` directly.
    """
    return make_task(TaskSpec("sine", {"freq": float(freq)}),
                     T=T, batch=batch, input_dim=input_dim,
                     seed=seed, device=device)


# ── Wake phase ───────────────────────────────────────────────────────────────

def wake_phase(model, inputs, targets, steps: int = 200,
               lr: float = 3e-3, ach_gate: float = 1.0,
               grad_clip: float = 1.0) -> list[float]:
    """Standard BPTT training on one task.

    ach_gate scales the effective learning rate (1.0 = full plasticity).
    Returns per-step mean loss.
    """
    model.train()
    model.unfreeze_reservoir()
    opt = torch.optim.Adam(model.parameters(), lr=lr * ach_gate)
    loss_fn = nn.MSELoss()
    device = next(model.parameters()).device
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        h = model.init_hidden(inputs.shape[1], device)
        total = torch.tensor(0.0, device=device)
        for t in range(inputs.shape[0]):
            out, h = model(inputs[t], h, feedback_detach=True)
            total = total + loss_fn(out, targets[t])
            h = h.detach()
        (total / inputs.shape[0]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(total.item() / inputs.shape[0])
    return losses


# ── Sleep phase ──────────────────────────────────────────────────────────────

def sleep_phase(model, sleep_steps: int = 600, eta: float = 0.01,
                decay: float = 0.001, ach_gate: float = 0.3, device=None) -> dict:
    """Hebbian consolidation driven by the network's own limit-cycle dynamics.

    Rule: ΔW_rec = η_eff · (H_c^T H_c) / T − η_eff · decay · W_rec
    where H_c is the mean-centred idle trajectory. If the post-update SR
    falls below 1.0, revert (would collapse the limit cycle).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    eta_eff = eta * ach_gate

    h = model.init_hidden(1, device)
    states = []
    with torch.no_grad():
        for _ in range(sleep_steps):
            _, h = model(None, h)
            states.append(h[0])

    H = torch.stack(states)                    # (T, hidden_dim)
    H_c = H - H.mean(0)
    hebbian = (H_c.T @ H_c) / sleep_steps

    with torch.no_grad():
        delta_W = eta_eff * hebbian - eta_eff * decay * model.W_rec.data
        model.W_rec.data += delta_W

    post_sr = model.spectral_radius()
    reverted = False
    if post_sr < 1.0:
        with torch.no_grad():
            model.W_rec.data -= delta_W
        post_sr = model.spectral_radius()
        reverted = True

    return {"post_sr": post_sr,
            "delta_W_norm": delta_W.norm().item(),
            "reverted": reverted}


# ── Idle trajectory ──────────────────────────────────────────────────────────

def run_idle(model, steps: int = 300, device=None) -> np.ndarray:
    """Collect an idle hidden-state trajectory of shape (steps, hidden_dim)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    h = model.init_hidden(1, device)
    states = []
    with torch.no_grad():
        for _ in range(steps):
            _, h = model(None, h)
            states.append(h[0].cpu().numpy())
    return np.array(states)


# ── Canonical stream training loop ───────────────────────────────────────────

def train_stream(model,
                 stream: list[TaskSpec],
                 device,
                 with_sleep: bool = True,
                 wake_steps: int = 200,
                 sleep_steps: int = 600,
                 idle_steps: int = 300,
                 lr: float = 3e-3,
                 eta: float = 0.01,
                 decay: float = 0.001,
                 wake_ach: float = 1.0,
                 sleep_ach: float = 0.3,
                 grad_clip: float = 1.0,
                 batch: int = 4,
                 T_wake: int = 200,
                 input_dim: int = 16,
                 collect_snapshots: bool = True,
                 task_seed_fn: Optional[Callable[[int], int]] = None,
                 verbose: bool = False) -> dict:
    """
    Canonical wake/sleep loop over a task stream.

    Args:
        model:      LeakyRNN-like module (torch.nn.Module on `device`).
        stream:     list of TaskSpec — the sequence of tasks.
        device:     torch device.
        with_sleep: if False, skip the sleep phase after each wake phase.
        wake_steps: BPTT gradient updates per task.
        sleep_steps: idle steps used to build the Hebbian update.
        idle_steps: idle steps collected as the per-cycle trajectory.
        lr, eta, decay, wake_ach, sleep_ach, grad_clip: hyperparameters.
        batch, T_wake, input_dim: task-tensor dimensions.
        collect_snapshots: if True, record an idle trajectory after each
            wake+sleep cycle. If False, only record after the FINAL cycle.
        task_seed_fn: maps cycle index → task seed. Default: identity.
            The task seed drives per-channel phase / initial condition
            inside `make_task`; the model seed is set by the caller
            BEFORE constructing the model.
        verbose: log per-cycle progress.

    Returns:
        dict with:
            snapshots:      list[np.ndarray]  # per-cycle idle trajectories
            final_idle:     np.ndarray        # convenience alias
            wake_losses:    list[list[float]]
            sleep_diagnostics: list[dict]
            spectral_radius_trajectory: list[float]  # post-cycle SR
            stream_specs:   [spec.to_dict() for spec in stream]
    """
    if task_seed_fn is None:
        task_seed_fn = lambda i: i

    snapshots: list[np.ndarray] = []
    wake_losses: list[list[float]] = []
    sleep_diagnostics: list[dict] = []
    sr_trajectory: list[float] = []

    for i, spec in enumerate(stream):
        inputs, targets = make_task(spec, T=T_wake, batch=batch,
                                    input_dim=input_dim,
                                    seed=task_seed_fn(i), device=device)
        losses = wake_phase(model, inputs, targets, steps=wake_steps,
                            lr=lr, ach_gate=wake_ach, grad_clip=grad_clip)
        wake_losses.append(losses)

        if with_sleep:
            diag = sleep_phase(model, sleep_steps=sleep_steps, eta=eta,
                               decay=decay, ach_gate=sleep_ach, device=device)
        else:
            diag = {"post_sr": model.spectral_radius(),
                    "delta_W_norm": 0.0, "reverted": False, "skipped": True}
        sleep_diagnostics.append(diag)
        sr_trajectory.append(diag["post_sr"])

        take_snapshot = collect_snapshots or (i == len(stream) - 1)
        if take_snapshot:
            snapshots.append(run_idle(model, steps=idle_steps, device=device))
            if verbose:
                print(f"    cycle {i+1}/{len(stream)}: task={spec.label!r}  "
                      f"loss {losses[0]:.4f}→{losses[-1]:.4f}  "
                      f"SR={diag['post_sr']:.3f}",
                      flush=True)

    return {
        "snapshots":        snapshots,
        "final_idle":       snapshots[-1] if snapshots else None,
        "wake_losses":      wake_losses,
        "sleep_diagnostics": sleep_diagnostics,
        "spectral_radius_trajectory": sr_trajectory,
        "stream_specs":     [s.to_dict() for s in stream],
    }
