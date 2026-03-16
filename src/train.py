"""
Wake/sleep training loop and shared utilities.

wake_phase:  standard BPTT training on a task
sleep_phase: Hebbian consolidation driven by the network's own limit cycle
make_sine_task: generate sinusoidal prediction tasks
run_idle:    collect idle hidden state trajectory
"""

import torch
import torch.nn as nn
import numpy as np


# ── Task generation ───────────────────────────────────────────────────────────

def make_sine_task(freq: float, T: int = 200, batch: int = 4,
                   input_dim: int = 16, seed: int = 0, device=None):
    """Sinusoidal prediction task: predict next timestep of a multi-channel sine."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    t = torch.linspace(0, 2 * np.pi * freq, T)
    phases = torch.rand(input_dim, generator=rng) * 2 * np.pi
    signal = torch.sin(t.unsqueeze(-1) + phases.unsqueeze(0))
    signal = signal.unsqueeze(1).expand(T, batch, input_dim)
    if device:
        signal = signal.to(device)
    return signal[:-1], signal[1:]  # inputs, targets


# ── Wake phase ────────────────────────────────────────────────────────────────

def wake_phase(model, inputs, targets, steps: int = 200,
               lr: float = 3e-3, ach_gate: float = 1.0):
    """
    Standard BPTT training.
    ach_gate scales the learning rate (1.0 = full plasticity, <1 = gated).
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(total.item() / inputs.shape[0])
    return losses


# ── Sleep phase ───────────────────────────────────────────────────────────────

def sleep_phase(model, sleep_steps: int = 600, eta: float = 0.01,
                decay: float = 0.001, ach_gate: float = 0.3, device=None):
    """
    Hebbian consolidation driven by the network's own limit cycle dynamics.

    Rule: ΔW_rec = η_eff · (H_c^T H_c) / T  −  η_eff · decay · W_rec
    where H_c = H − mean(H) is the mean-centred idle trajectory.

    No external input. No stored data. The limit cycle IS the replay signal.
    Optimal config from Stage 3C sweep: eta=0.01, decay=0.001.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    eta_eff = eta * ach_gate

    # collect idle trajectory
    h = model.init_hidden(1, device)
    states = []
    with torch.no_grad():
        for _ in range(sleep_steps):
            _, h = model(None, h)
            states.append(h[0])

    H = torch.stack(states)                    # (T, hidden_dim)
    H_c = H - H.mean(0)                        # mean-centre
    hebbian = (H_c.T @ H_c) / sleep_steps      # (hidden_dim, hidden_dim)

    with torch.no_grad():
        delta_W = eta_eff * hebbian - eta_eff * decay * model.W_rec.data
        model.W_rec.data += delta_W

    post_sr = model.spectral_radius()
    if post_sr < 1.0:
        # SR collapsed — revert (limit cycle would die)
        with torch.no_grad():
            model.W_rec.data -= delta_W
        post_sr = model.spectral_radius()

    return {"post_sr": post_sr, "delta_W_norm": delta_W.norm().item()}


# ── Idle trajectory ───────────────────────────────────────────────────────────

def run_idle(model, steps: int = 300, device=None) -> np.ndarray:
    """Collect idle hidden state trajectory. Returns (steps, hidden_dim) array."""
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


# ── Divergence metrics ────────────────────────────────────────────────────────

def subspace_angle(sa: np.ndarray, sb: np.ndarray, n: int = 3) -> float:
    """
    Subspace angle between top-n principal components of two idle trajectories.
    Rotation-invariant measure of representational geometry divergence.
    """
    _, _, Va = np.linalg.svd(sa - sa.mean(0), full_matrices=False)
    _, _, Vb = np.linalg.svd(sb - sb.mean(0), full_matrices=False)
    sv = np.clip(np.linalg.svd(Va[:n] @ Vb[:n].T, compute_uv=False), -1, 1)
    return float(np.arccos(sv).mean() * 180 / np.pi)


def decoder_accuracy(states_by_label: dict) -> float:
    """Linear decoder accuracy on idle hidden states, keyed by label."""
    labels = sorted(states_by_label.keys())
    X = np.vstack([states_by_label[l][-150:] for l in labels])
    y = np.concatenate([[i] * 150 for i in range(len(labels))])
    W, _, _, _ = np.linalg.lstsq(X, np.eye(len(labels))[y], rcond=None)
    return float(((X @ W).argmax(1) == y).mean())
