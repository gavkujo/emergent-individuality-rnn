"""
LeakyRNN with self-feedback loop.

Architecture:
    x(t+1) = (1 - 1/τ) · x(t) + (1/τ) · tanh(W_rec·x(t) + W_in·u(t) + b)
    y(t)   = W_out · x(t)
    u(t)   = y(t-1)   [self-feedback during idle]

W_rec is trained (not frozen). W_in is fixed random.
Spectral radius of W_rec is initialised to target_sr and rises during training
into the supercritical (edge-of-chaos) regime.

See architecture.md for design justification.
"""

import torch
import torch.nn as nn
import numpy as np


class LeakyRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 tau: float = 5.0, alpha: float = 0.8, target_sr: float = 0.95):
        """
        Args:
            input_dim:  external input dimension
            hidden_dim: recurrent hidden state dimension
            output_dim: readout dimension (also feedback dimension)
            tau:        memory timescale — higher = slower dynamics, longer memory
            alpha:      input blend during wake (1=pure external, 0=pure feedback)
            target_sr:  initial spectral radius of W_rec
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        self.alpha = alpha
        self.leak = 1.0 / tau

        # recurrent weights — scaled to target spectral radius
        W_rec = torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        with torch.no_grad():
            eigvals = torch.linalg.eigvals(W_rec.cpu())
            current_sr = eigvals.abs().max().item()
            if current_sr > 0:
                W_rec = W_rec * (target_sr / current_sr)
        self.W_rec = nn.Parameter(W_rec)

        # input weights — fixed random (not trained)
        self.register_buffer("W_in", torch.randn(hidden_dim, input_dim) * 0.1)

        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.W_out = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)

    def forward(self, x, h, feedback_detach: bool = True):
        """
        Args:
            x: (batch, input_dim) or None for idle operation
            h: (batch, hidden_dim) current hidden state
            feedback_detach: detach feedback from computation graph (recommended)

        Returns:
            out: (batch, output_dim)
            h_new: (batch, hidden_dim)
        """
        fb = h @ self.W_out.T
        if feedback_detach:
            fb = fb.detach()

        if x is None:
            # idle: only feedback path active
            inp = fb @ self.W_in.T
        else:
            inp = self.alpha * (x @ self.W_in.T) + (1.0 - self.alpha) * (fb @ self.W_in.T)

        h_new = (1.0 - self.leak) * h + self.leak * torch.tanh(inp + h @ self.W_rec.T + self.bias)
        return h_new @ self.W_out.T, h_new

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.randn(batch_size, self.hidden_dim, device=device) * 0.1

    def spectral_radius(self) -> float:
        return torch.linalg.eigvals(self.W_rec.data.cpu()).abs().max().item()

    def freeze_reservoir(self):
        self.W_rec.requires_grad_(False)

    def unfreeze_reservoir(self):
        self.W_rec.requires_grad_(True)
