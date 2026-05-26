"""
Baseline architectures with the same self-feedback loop as LeakyRNN.

All baselines implement the same interface:
    forward(x, h, feedback_detach=True) -> (out, h_new)
    init_hidden(batch_size, device) -> hidden state tensor
    spectral_radius() -> float
    freeze_reservoir() / unfreeze_reservoir()

Shared design:
    - Output y(t) = W_out · h_relevant(t)
    - Idle (x=None): u(t) = y(t-1) fed back through W_in
    - Wake: u(t) = α · x_ext + (1-α) · y(t-1), then through W_in
    - W_in: fixed random buffer, scale 0.1 (not trained)
    - W_out: trainable, scale 0.1
    - hidden_dim, input_dim, output_dim match LeakyRNN config

Architectures:
    VanillaRNN — h = tanh(W_rec·h + inp + b)
    GatedRNN   — wraps nn.GRUCell
    LSTM       — wraps nn.LSTMCell, packs (h,c) state into single tensor
    FrozenESN  — same as LeakyRNN but W_rec frozen (only W_out trains)
"""

import torch
import torch.nn as nn
import numpy as np


def _scale_to_sr(W: torch.Tensor, target_sr: float) -> torch.Tensor:
    """Scale a square matrix to a target spectral radius."""
    eigvals = torch.linalg.eigvals(W.cpu())
    current_sr = eigvals.abs().max().item()
    if current_sr > 0:
        return W * (target_sr / current_sr)
    return W


# ── Vanilla RNN ───────────────────────────────────────────────────────────────

class VanillaRNN(nn.Module):
    """Plain RNN: h = tanh(W_rec·h + W_in·u + b). No leak, no gating."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 target_sr: float = 0.95, alpha: float = 0.8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha

        W_rec = torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        with torch.no_grad():
            W_rec = _scale_to_sr(W_rec, target_sr)
        self.W_rec = nn.Parameter(W_rec)
        self.register_buffer("W_in", torch.randn(hidden_dim, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.W_out = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)

    def forward(self, x, h, feedback_detach: bool = True):
        fb = h @ self.W_out.T
        if feedback_detach:
            fb = fb.detach()
        if x is None:
            inp = fb @ self.W_in.T
        else:
            inp = self.alpha * (x @ self.W_in.T) + (1.0 - self.alpha) * (fb @ self.W_in.T)
        h_new = torch.tanh(inp + h @ self.W_rec.T + self.bias)
        return h_new @ self.W_out.T, h_new

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.randn(batch_size, self.hidden_dim, device=device) * 0.1

    def spectral_radius(self) -> float:
        return torch.linalg.eigvals(self.W_rec.data.cpu()).abs().max().item()

    def freeze_reservoir(self):
        self.W_rec.requires_grad_(False)

    def unfreeze_reservoir(self):
        self.W_rec.requires_grad_(True)


# ── Gated RNN (GRU wrapper) ───────────────────────────────────────────────────

class GatedRNN(nn.Module):
    """GRU with the shared self-feedback loop."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 alpha: float = 0.8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.register_buffer("W_in", torch.randn(hidden_dim, input_dim) * 0.1)
        self.W_out = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)

    def forward(self, x, h, feedback_detach: bool = True):
        fb = h @ self.W_out.T
        if feedback_detach:
            fb = fb.detach()
        if x is None:
            u = fb
        else:
            u = self.alpha * x + (1.0 - self.alpha) * fb
        h_new = self.cell(u, h)
        return h_new @ self.W_out.T, h_new

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.randn(batch_size, self.hidden_dim, device=device) * 0.1

    def spectral_radius(self) -> float:
        # GRU has 3 gate weight matrices; report SR of the candidate-state weights
        # (the part most analogous to W_rec in a vanilla RNN).
        W = self.cell.weight_hh.data
        # weight_hh is (3*hidden, hidden); rows: reset, update, new
        W_new = W[2 * self.hidden_dim : 3 * self.hidden_dim].cpu()
        return torch.linalg.eigvals(W_new).abs().max().item()

    def freeze_reservoir(self):
        for p in self.cell.parameters():
            p.requires_grad_(False)

    def unfreeze_reservoir(self):
        for p in self.cell.parameters():
            p.requires_grad_(True)


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMNet(nn.Module):
    """LSTM with the shared self-feedback loop. Hidden state packs (h, c)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 alpha: float = 0.8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.cell = nn.LSTMCell(input_dim, hidden_dim)
        self.register_buffer("W_in", torch.randn(hidden_dim, input_dim) * 0.1)
        self.W_out = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)

    def _split(self, h_packed: torch.Tensor):
        # h_packed: (batch, 2*hidden_dim) → (h, c) tuple
        return h_packed[:, :self.hidden_dim], h_packed[:, self.hidden_dim:]

    def _pack(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, c], dim=-1)

    def forward(self, x, h_packed, feedback_detach: bool = True):
        h, c = self._split(h_packed)
        fb = h @ self.W_out.T
        if feedback_detach:
            fb = fb.detach()
        if x is None:
            u = fb
        else:
            u = self.alpha * x + (1.0 - self.alpha) * fb
        h_new, c_new = self.cell(u, (h, c))
        return h_new @ self.W_out.T, self._pack(h_new, c_new)

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.randn(batch_size, 2 * self.hidden_dim, device=device) * 0.1

    def spectral_radius(self) -> float:
        # weight_hh: (4*hidden, hidden); rows: input, forget, cell, output
        # report SR of the cell-gate (g) weights, most analogous to W_rec.
        W = self.cell.weight_hh.data
        W_g = W[2 * self.hidden_dim : 3 * self.hidden_dim].cpu()
        return torch.linalg.eigvals(W_g).abs().max().item()

    def freeze_reservoir(self):
        for p in self.cell.parameters():
            p.requires_grad_(False)

    def unfreeze_reservoir(self):
        for p in self.cell.parameters():
            p.requires_grad_(True)


# ── Frozen-reservoir ESN ──────────────────────────────────────────────────────

class FrozenESN(nn.Module):
    """
    LeakyRNN with W_rec frozen (only W_out trained).
    The classical Echo State Network setup: random fixed reservoir, linear readout.
    Spectral radius is set at init and never changes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 tau: float = 5.0, alpha: float = 0.8, target_sr: float = 0.95):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        self.alpha = alpha
        self.leak = 1.0 / tau

        W_rec = torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        with torch.no_grad():
            W_rec = _scale_to_sr(W_rec, target_sr)
        # W_rec is a buffer — not a parameter, never trained
        self.register_buffer("W_rec", W_rec)
        self.register_buffer("W_in", torch.randn(hidden_dim, input_dim) * 0.1)
        self.register_buffer("bias", torch.zeros(hidden_dim))
        self.W_out = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)

    def forward(self, x, h, feedback_detach: bool = True):
        fb = h @ self.W_out.T
        if feedback_detach:
            fb = fb.detach()
        if x is None:
            inp = fb @ self.W_in.T
        else:
            inp = self.alpha * (x @ self.W_in.T) + (1.0 - self.alpha) * (fb @ self.W_in.T)
        h_new = (1.0 - self.leak) * h + self.leak * torch.tanh(inp + h @ self.W_rec.T + self.bias)
        return h_new @ self.W_out.T, h_new

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.randn(batch_size, self.hidden_dim, device=device) * 0.1

    def spectral_radius(self) -> float:
        return torch.linalg.eigvals(self.W_rec.cpu()).abs().max().item()

    def freeze_reservoir(self):
        # already frozen
        pass

    def unfreeze_reservoir(self):
        # ESN keeps the reservoir frozen by definition. No-op.
        pass
