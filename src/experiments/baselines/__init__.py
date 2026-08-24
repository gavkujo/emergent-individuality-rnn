"""
Baseline architecture comparison.

Compares LeakyRNN against VanillaRNN, GatedRNN (GRU), LSTM, and FrozenESN on:
    1. Idle richness (untrained and trained)
    2. Decoder accuracy on idle states across 4 training frequencies

All baselines share the same self-feedback loop, hidden_dim=128, identical
training (Adam, lr=3e-3, 300 steps on freq=3.0 sinusoid), and identical idle
measurement (300 steps).
"""

import torch
import numpy as np

from src.model import LeakyRNN
from src.baselines import VanillaRNN, GatedRNN, LSTMNet, FrozenESN
from src.train import wake_phase, run_idle, make_sine_task, decoder_accuracy


NAME = "baseline_comparison"
DESCRIPTION = ("Fair architecture comparison: LeakyRNN vs VanillaRNN, GRU, "
               "LSTM, FrozenESN. Idle richness and 4-freq decoder accuracy.")
DETAILS = ("v2: decoder_accuracy switched from in-sample lstsq to held-out "
           "5-fold ridge CV (returns dict with mean, std, in_sample, "
           "ridge_lambda). The previous v1 numbers (1.000 across all "
           "architectures) were classifier capacity, not network encoding.")
VERSION = "v2"

INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 16, 128, 16
TRAIN_STEPS, IDLE_STEPS = 300, 300
SEED = 42


def _archs():
    return {
        "LeakyRNN":   lambda: LeakyRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
                                       tau=2.0, target_sr=1.05),
        "VanillaRNN": lambda: VanillaRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
                                         target_sr=0.95),
        "GatedRNN":   lambda: GatedRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
        "LSTM":       lambda: LSTMNet(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
        "FrozenESN":  lambda: FrozenESN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
                                        tau=5.0, target_sr=0.95),
    }


def _make(ctor, device):
    torch.manual_seed(SEED)
    return ctor().to(device)


def _measure_richness(model, device, train: bool):
    if train:
        wake_phase(model, *make_sine_task(3.0, device=device), steps=TRAIN_STEPS)
    states = run_idle(model, steps=IDLE_STEPS, device=device)
    return {
        "richness": float(states.std(0).mean()),
        "activity": float(np.abs(states).mean()),
        "spectral_radius": model.spectral_radius(),
    }


def _measure_decoder(ctor, device):
    states_by_freq = {}
    for freq in [1.0, 2.0, 4.0, 8.0]:
        m = _make(ctor, device)
        wake_phase(m, *make_sine_task(freq, device=device), steps=TRAIN_STEPS)
        states_by_freq[freq] = run_idle(m, steps=IDLE_STEPS, device=device)
    return decoder_accuracy(states_by_freq)


def run(device) -> dict:
    out = {"architectures": {}, "config": {
        "hidden_dim": HIDDEN_DIM, "input_dim": INPUT_DIM, "output_dim": OUTPUT_DIM,
        "train_steps": TRAIN_STEPS, "idle_steps": IDLE_STEPS, "seed": SEED,
        "device": str(device),
    }}

    for name, ctor in _archs().items():
        print(f"\n── {name} ──")

        untrained = _measure_richness(_make(ctor, device), device, train=False)
        print(f"  untrained: richness={untrained['richness']:.4f} "
              f"activity={untrained['activity']:.4f} "
              f"SR={untrained['spectral_radius']:.3f}")

        trained = _measure_richness(_make(ctor, device), device, train=True)
        print(f"  trained:   richness={trained['richness']:.4f} "
              f"activity={trained['activity']:.4f} "
              f"SR={trained['spectral_radius']:.3f}")

        print("  decoder accuracy on idle states (4 freqs)...")
        dec = _measure_decoder(ctor, device)
        print(f"  decoder acc: {dec['mean']:.3f} ± {dec['std']:.3f} "
              f"(in-sample {dec['in_sample']:.3f})")

        out["architectures"][name] = {
            "untrained": untrained,
            "trained": trained,
            "decoder": dec,
        }

    return out
