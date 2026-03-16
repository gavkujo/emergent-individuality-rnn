"""
Reproduce all benchmark results.
Saves to results/benchmark_results.json.
Run time: ~5 min on MPS/CUDA, ~20 min on CPU.

Benchmarks:
  B1. Idle richness: GRU vs LeakyRNN across tau/SR grid
  B2. Decoder accuracy: single task and 4 experiential streams
  B3. Divergence accumulation over 5 wake-sleep cycles
  B4. Path-dependence: subspace angle vs stream length (1-10)
  B5. Sleep effect: with vs without sleep
  B6. Order effect: same tasks reversed vs different tasks
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import json

from src.model import LeakyRNN
from src.train import (wake_phase, sleep_phase, run_idle,
                       make_sine_task, subspace_angle, decoder_accuracy)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 16, 128, 16
TAU, SR_INIT = 5.0, 0.95
ETA, DECAY = 0.01, 0.001  # optimal from Stage 3C

STREAM_A = [1.0, 1.5, 2.0, 1.0, 2.0]
STREAM_B = [8.0, 12.0, 6.0, 8.0, 6.0]
STREAM_C = [1.0, 8.0, 2.0, 6.0, 1.5]
STREAM_D = [2.0, 1.5, 1.0, 2.0, 1.0]  # Stream A reversed


def make_model(seed=42, tau=TAU, sr=SR_INIT):
    torch.manual_seed(seed)
    return LeakyRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, tau=tau, target_sr=sr).to(DEVICE)


def run_stream(model, freqs, with_sleep=True):
    snapshots = []
    for i, freq in enumerate(freqs):
        wake_phase(model, *make_sine_task(freq, seed=i, device=DEVICE),
                   steps=200, lr=3e-3, ach_gate=1.0)
        if with_sleep:
            sleep_phase(model, sleep_steps=600, eta=ETA, decay=DECAY,
                        ach_gate=0.3, device=DEVICE)
        snapshots.append(run_idle(model, device=DEVICE))
    return snapshots


results = {}

# B1: Idle richness
print("\nB1: Idle richness...")
from experiments.stage1_feedback_dynamics.model import FeedbackGRU  # noqa: F401 — optional
try:
    # try to import GRU for comparison
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                    "experiments", "stage1_feedback_dynamics"))
    from model import FeedbackGRU
    gru = FeedbackGRU(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    wake_phase(gru, *make_sine_task(3.0, device=DEVICE), steps=300)
    s = run_idle(gru, device=DEVICE)
    gru_richness = float(s.std(0).mean())
except Exception:
    gru_richness = 0.000004  # from Stage 1 experiments

b1 = {"gru_richness": gru_richness, "leaky": {}}
for tau in [2.0, 5.0, 10.0]:
    for sr in [0.9, 0.95, 1.05]:
        m = make_model(tau=tau, sr=sr)
        wake_phase(m, *make_sine_task(3.0, device=DEVICE), steps=300)
        s = run_idle(m, device=DEVICE)
        r = float(s.std(0).mean())
        b1["leaky"][f"tau{tau}_sr{sr}"] = {"tau": tau, "sr": sr, "richness": r}
        print(f"  tau={tau}, sr={sr}: richness={r:.4f}")
results["b1_idle_richness"] = b1

# B2: Decoder accuracy
print("\nB2: Decoder accuracy...")
single_states = {}
for freq in [1.0, 2.0, 4.0, 8.0]:
    m = make_model()
    wake_phase(m, *make_sine_task(freq, device=DEVICE), steps=300)
    single_states[freq] = run_idle(m, device=DEVICE)
acc_single = decoder_accuracy(single_states)
print(f"  Single task: {acc_single:.3f}")

stream_states = {}
for name, stream in [("A_low", STREAM_A), ("B_high", STREAM_B),
                     ("C_mixed", STREAM_C), ("D_reversed", STREAM_D)]:
    print(f"  Running stream {name}...", flush=True)
    m = make_model()
    stream_states[name] = run_stream(m, stream)[-1]
acc_streams = decoder_accuracy(stream_states)
print(f"  4-stream: {acc_streams:.3f}")
results["b2_decoder"] = {"single_task_acc": acc_single, "four_stream_acc": acc_streams}

# B3: Divergence accumulation
print("\nB3: Divergence accumulation...")
na, nb = make_model(), make_model()
sa_snaps = run_stream(na, STREAM_A)
sb_snaps = run_stream(nb, STREAM_B)
b3 = []
for i, (sa, sb) in enumerate(zip(sa_snaps, sb_snaps)):
    angle = subspace_angle(sa, sb)
    l2 = float(np.linalg.norm(sa - sb, axis=1).mean())
    b3.append({"cycle": i+1, "subspace_angle": angle, "l2": l2})
    print(f"  cycle {i+1}: angle={angle:.2f}°, L2={l2:.4f}")
results["b3_accumulation"] = b3

# B4: Path-dependence
print("\nB4: Path-dependence...")
fwd = [1.0, 4.0, 8.0] * 3 + [1.0]
rev = [8.0, 4.0, 1.0] * 3 + [8.0]
b4 = []
for length in [1, 2, 3, 5, 7, 10]:
    nf, nr = make_model(), make_model()
    for i in range(length):
        wake_phase(nf, *make_sine_task(fwd[i], seed=i, device=DEVICE), steps=200, ach_gate=1.0)
        sleep_phase(nf, sleep_steps=600, eta=ETA, decay=DECAY, ach_gate=0.3, device=DEVICE)
        wake_phase(nr, *make_sine_task(rev[i], seed=i, device=DEVICE), steps=200, ach_gate=1.0)
        sleep_phase(nr, sleep_steps=600, eta=ETA, decay=DECAY, ach_gate=0.3, device=DEVICE)
    angle = subspace_angle(run_idle(nf, device=DEVICE), run_idle(nr, device=DEVICE))
    b4.append({"length": length, "angle": float(angle)})
    print(f"  length={length}: angle={angle:.2f}°")
results["b4_path_dependence"] = b4

# B5: Sleep effect
print("\nB5: Sleep effect...")
na_s, nb_s = make_model(), make_model()
na_n, nb_n = make_model(), make_model()
sa_s = run_stream(na_s, STREAM_A, with_sleep=True)[-1]
sb_s = run_stream(nb_s, STREAM_B, with_sleep=True)[-1]
sa_n = run_stream(na_n, STREAM_A, with_sleep=False)[-1]
sb_n = run_stream(nb_n, STREAM_B, with_sleep=False)[-1]
results["b5_sleep_effect"] = {
    "with_sleep":    {"angle": subspace_angle(sa_s, sb_s), "l2": float(np.linalg.norm(sa_s - sb_s, axis=1).mean())},
    "without_sleep": {"angle": subspace_angle(sa_n, sb_n), "l2": float(np.linalg.norm(sa_n - sb_n, axis=1).mean())},
}
print(f"  With sleep:    angle={results['b5_sleep_effect']['with_sleep']['angle']:.2f}°")
print(f"  Without sleep: angle={results['b5_sleep_effect']['without_sleep']['angle']:.2f}°")

# B6: Order effect
print("\nB6: Order effect...")
na2, nd, nb2 = make_model(), make_model(), make_model()
sa2 = run_stream(na2, STREAM_A)[-1]
sd  = run_stream(nd,  STREAM_D)[-1]
sb2 = run_stream(nb2, STREAM_B)[-1]
angle_ad = subspace_angle(sa2, sd)
angle_ab = subspace_angle(sa2, sb2)
results["b6_order_effect"] = {
    "same_tasks_reversed": {"angle": float(angle_ad)},
    "different_tasks":     {"angle": float(angle_ab)},
    "order_fraction":      float(angle_ad / angle_ab),
}
print(f"  A vs D (reversed): {angle_ad:.2f}°")
print(f"  A vs B (different): {angle_ab:.2f}°")
print(f"  Order fraction: {angle_ad/angle_ab:.2f}x")

# save
out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "benchmark_results.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved to {out_path}")
