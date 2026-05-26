"""
Full benchmark suite (B1-B6) for the LeakyRNN model.

  B1. Idle richness across tau/SR grid (LeakyRNN only;
      cross-architecture lives in baselines.py)
  B2. Decoder accuracy: single task and 4 experiential streams
  B3. Divergence accumulation over 5 wake-sleep cycles
  B4. Path-dependence: subspace angle vs stream length (1-10)
  B5. Sleep effect: with vs without sleep
  B6. Order effect: same tasks reversed vs different tasks
"""

import torch
import numpy as np

from src.model import LeakyRNN
from src.train import (wake_phase, sleep_phase, run_idle,
                       make_sine_task, subspace_angle, decoder_accuracy)


NAME = "benchmark"
DESCRIPTION = ("Full B1-B6 benchmark for LeakyRNN: idle richness, decoder "
               "accuracy, divergence accumulation, path-dependence, sleep effect, "
               "order effect.")
DETAILS = ("v2: replaces the inflated 'GRU 50,000x' framing of B1 with a clean "
           "LeakyRNN tau/SR sweep; cross-architecture moved to baselines.py.")
VERSION = "v2"

INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 16, 128, 16
TAU, SR_INIT = 5.0, 0.95
ETA, DECAY = 0.01, 0.001

STREAM_A = [1.0, 1.5, 2.0, 1.0, 2.0]
STREAM_B = [8.0, 12.0, 6.0, 8.0, 6.0]
STREAM_C = [1.0, 8.0, 2.0, 6.0, 1.5]
STREAM_D = [2.0, 1.5, 1.0, 2.0, 1.0]


def _make(device, seed=42, tau=TAU, sr=SR_INIT):
    torch.manual_seed(seed)
    return LeakyRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
                    tau=tau, target_sr=sr).to(device)


def _run_stream(model, freqs, device, with_sleep=True):
    snaps = []
    for i, freq in enumerate(freqs):
        wake_phase(model, *make_sine_task(freq, seed=i, device=device),
                   steps=200, lr=3e-3, ach_gate=1.0)
        if with_sleep:
            sleep_phase(model, sleep_steps=600, eta=ETA, decay=DECAY,
                        ach_gate=0.3, device=device)
        snaps.append(run_idle(model, device=device))
    return snaps


def run(device) -> dict:
    out = {}

    # B1
    print("\nB1: Idle richness (LeakyRNN tau/SR sweep)...")
    b1 = {}
    for tau in [2.0, 5.0, 10.0]:
        for sr in [0.9, 0.95, 1.05]:
            m = _make(device, tau=tau, sr=sr)
            wake_phase(m, *make_sine_task(3.0, device=device), steps=300)
            s = run_idle(m, device=device)
            r = float(s.std(0).mean())
            b1[f"tau{tau}_sr{sr}"] = {"tau": tau, "sr": sr, "richness": r,
                                       "activity": float(np.abs(s).mean())}
            print(f"  tau={tau}, sr={sr}: richness={r:.4f}")
    out["b1_idle_richness_leaky"] = b1

    # B2
    print("\nB2: Decoder accuracy...")
    single_states = {}
    for freq in [1.0, 2.0, 4.0, 8.0]:
        m = _make(device)
        wake_phase(m, *make_sine_task(freq, device=device), steps=300)
        single_states[freq] = run_idle(m, device=device)
    acc_single = decoder_accuracy(single_states)
    print(f"  Single task: {acc_single:.3f}")

    stream_states = {}
    for name, stream in [("A_low", STREAM_A), ("B_high", STREAM_B),
                         ("C_mixed", STREAM_C), ("D_reversed", STREAM_D)]:
        print(f"  Running stream {name}...", flush=True)
        m = _make(device)
        stream_states[name] = _run_stream(m, stream, device)[-1]
    acc_streams = decoder_accuracy(stream_states)
    print(f"  4-stream: {acc_streams:.3f}")
    out["b2_decoder"] = {"single_task_acc": acc_single,
                         "four_stream_acc": acc_streams,
                         "chance_single": 0.25, "chance_streams": 0.25}

    # B3
    print("\nB3: Divergence accumulation...")
    na, nb = _make(device), _make(device)
    sa_snaps = _run_stream(na, STREAM_A, device)
    sb_snaps = _run_stream(nb, STREAM_B, device)
    b3 = []
    for i, (sa, sb) in enumerate(zip(sa_snaps, sb_snaps)):
        angle = subspace_angle(sa, sb)
        l2 = float(np.linalg.norm(sa - sb, axis=1).mean())
        b3.append({"cycle": i+1, "freq_a": STREAM_A[i], "freq_b": STREAM_B[i],
                   "subspace_angle": angle, "l2": l2})
        print(f"  cycle {i+1}: angle={angle:.2f}°, L2={l2:.4f}")
    out["b3_accumulation"] = b3

    # B4
    print("\nB4: Path-dependence...")
    fwd = [1.0, 4.0, 8.0] * 3 + [1.0]
    rev = [8.0, 4.0, 1.0] * 3 + [8.0]
    b4 = []
    for length in [1, 2, 3, 5, 7, 10]:
        nf, nr = _make(device), _make(device)
        for i in range(length):
            wake_phase(nf, *make_sine_task(fwd[i], seed=i, device=device),
                       steps=200, ach_gate=1.0)
            sleep_phase(nf, sleep_steps=600, eta=ETA, decay=DECAY,
                        ach_gate=0.3, device=device)
            wake_phase(nr, *make_sine_task(rev[i], seed=i, device=device),
                       steps=200, ach_gate=1.0)
            sleep_phase(nr, sleep_steps=600, eta=ETA, decay=DECAY,
                        ach_gate=0.3, device=device)
        angle = subspace_angle(run_idle(nf, device=device),
                               run_idle(nr, device=device))
        b4.append({"length": length, "angle": float(angle)})
        print(f"  length={length}: angle={angle:.2f}°")
    out["b4_path_dependence"] = b4

    # B5
    print("\nB5: Sleep effect...")
    na_s, nb_s = _make(device), _make(device)
    na_n, nb_n = _make(device), _make(device)
    sa_s = _run_stream(na_s, STREAM_A, device, with_sleep=True)[-1]
    sb_s = _run_stream(nb_s, STREAM_B, device, with_sleep=True)[-1]
    sa_n = _run_stream(na_n, STREAM_A, device, with_sleep=False)[-1]
    sb_n = _run_stream(nb_n, STREAM_B, device, with_sleep=False)[-1]
    out["b5_sleep_effect"] = {
        "with_sleep": {"angle": subspace_angle(sa_s, sb_s),
                       "l2": float(np.linalg.norm(sa_s - sb_s, axis=1).mean())},
        "without_sleep": {"angle": subspace_angle(sa_n, sb_n),
                          "l2": float(np.linalg.norm(sa_n - sb_n, axis=1).mean())},
    }
    print(f"  With sleep:    angle={out['b5_sleep_effect']['with_sleep']['angle']:.2f}°")
    print(f"  Without sleep: angle={out['b5_sleep_effect']['without_sleep']['angle']:.2f}°")

    # B6
    print("\nB6: Order effect...")
    na2, nd, nb2 = _make(device), _make(device), _make(device)
    sa2 = _run_stream(na2, STREAM_A, device)[-1]
    sd  = _run_stream(nd,  STREAM_D, device)[-1]
    sb2 = _run_stream(nb2, STREAM_B, device)[-1]
    angle_ad = subspace_angle(sa2, sd)
    angle_ab = subspace_angle(sa2, sb2)
    out["b6_order_effect"] = {
        "same_tasks_reversed": {"angle": float(angle_ad)},
        "different_tasks":     {"angle": float(angle_ab)},
        "order_fraction":      float(angle_ad / angle_ab),
    }
    print(f"  A vs D (reversed): {angle_ad:.2f}°")
    print(f"  A vs B (different): {angle_ab:.2f}°")
    print(f"  Order fraction: {angle_ad/angle_ab:.2f}x")

    return out
