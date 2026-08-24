"""
Anchored-last-cycle order-effect test — isolate middle-order from recency.

Motivation. `order_effect_hardened_v2` showed that under random-permutation
streams, angle(same-content-different-order) ≈ angle(different-content-same-
distribution) ≈ 72°. Two interpretations were consistent with the data:

    (a) Recency dominates: the idle state is determined by the last task,
        and both same-permutation-diff-permutation AND same-perm-diff-content
        randomise the last task equally, so both angles are equal and large.
    (b) Everything matters equally: order and content contribute
        equivalently to individuality, each producing near-orthogonal
        subspaces on its own.

To distinguish, force the LAST task to be identical across all streams and
only vary the first K-1 tasks.

Design (per seed):

    - anchor_freq  ~ log-uniform [0.5, 20]  (shared last task)
    - base_freqs   ~ log-uniform [0.5, 20], K-1 of them
    - alt_freqs    ~ log-uniform [0.5, 20], K-1 fresh ones
    - perm_A       = random permutation of {0..K-2}
    - perm_B       = different random permutation of {0..K-2}

    S_A = [base_freqs[perm_A[0]], ..., base_freqs[perm_A[K-2]], anchor_freq]
    S_B = [base_freqs[perm_B[0]], ..., base_freqs[perm_B[K-2]], anchor_freq]
    T   = [alt_freqs [perm_A[0]], ..., alt_freqs [perm_A[K-2]], anchor_freq]

    S_A vs S_B: same content in first K-1, DIFFERENT order, SAME anchor
                → isolates middle-order effect (holding both content and
                  recency fixed).
    S_A vs T:   DIFFERENT content in first K-1, SAME order, SAME anchor
                → isolates middle-content effect (holding both order slot
                  and recency fixed).

Verdicts:

    angle(S_A, S_B) ≈ 0:
        Middle order does not matter. Together with v2, this pins the
        entire "individuality-from-experience" signature on the LAST
        task. The paper's order claim is dead.

    angle(S_A, S_B) large, comparable to v2's ~72°:
        Middle order contributes substantially to individuality. The
        paper has a defensible order claim, just narrower than
        originally framed.

    angle(S_A, S_B) intermediate, similar magnitude to angle(S_A, T):
        Middle order and middle content contribute equivalently.
        Individuality is driven by "any variation," not specifically
        by ORDER.

Also computes ratio angle(S_A, S_B) / angle(S_A, T) for direct
comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.harness import run_seeded
from src.metrics import (bootstrap_ci_mean, subspace_angle, l2_divergence)
from src.model import LeakyRNN
from src.tasks import TaskSpec
from src.train import train_stream


NAME = "order_effect_anchored"
DESCRIPTION = ("Anchored-last-cycle test isolating middle-order from "
               "recency. Per seed: 3 streams sharing the SAME last task; "
               "same-content-different-order vs different-content-same-order. "
               "n=30 seeds, bootstrap CI on angles and ratio.")
DETAILS = ("v1: LeakyRNN (hidden_dim=256, τ=5, target_sr=0.95, α=0.8). "
           "Stream length L=5, last-cycle freq shared across all three "
           "streams per seed. Middle 4 freqs sampled log-uniform "
           "[0.5, 20]; two permutations of the same 4 (S_A, S_B), plus "
           "T = 4 fresh freqs in the same permutation as S_A. Wake+sleep, "
           "canonical hyperparameters. Bootstrap 2000 resamples for CI.")
VERSION = "v1"


CONFIG: dict[str, Any] = {
    "hidden_dim": 256,
    "input_dim":  16,
    "output_dim": 16,
    "tau":        5.0,
    "target_sr":  0.95,
    "alpha":      0.8,
    "task_family":  "sine",
    "stream_length": 5,
    "freq_log_lo": 0.5,
    "freq_log_hi": 20.0,
    "wake_steps":  200,
    "wake_T":      200,
    "wake_batch":  4,
    "wake_lr":     3e-3,
    "wake_ach":    1.0,
    "sleep_steps": 600,
    "eta":         0.01,
    "decay":       0.001,
    "sleep_ach":   0.3,
    "grad_clip":   1.0,
    "idle_steps":  400,
    "bootstrap_resamples": 2000,
    "bootstrap_ci":        0.95,
    "n_seeds":     30,
    # Sub-seed offsets for reproducible stream construction
    "base_freqs_offset":   800000,
    "alt_freqs_offset":    900000,
    "anchor_freq_offset": 1000000,
    "perm_A_offset":      1100000,
    "perm_B_offset":      1200000,
}


# ── Stream construction ─────────────────────────────────────────────────────

def _sample_freqs(K: int, seed: int, lo: float, hi: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=K))


def _spec(freq: float) -> TaskSpec:
    return TaskSpec("sine", {"freq": float(freq)}, f"sine_{freq:.2f}Hz")


def _fresh_leakyrnn(seed: int, config: dict, device):
    torch.manual_seed(seed)
    return LeakyRNN(config["input_dim"], config["hidden_dim"],
                    config["output_dim"], tau=config["tau"],
                    alpha=config["alpha"],
                    target_sr=config["target_sr"]).to(device)


def _train(model, stream, config, device):
    return train_stream(
        model, stream, device, with_sleep=True,
        wake_steps=config["wake_steps"], sleep_steps=config["sleep_steps"],
        idle_steps=config["idle_steps"], lr=config["wake_lr"],
        eta=config["eta"], decay=config["decay"],
        wake_ach=config["wake_ach"], sleep_ach=config["sleep_ach"],
        grad_clip=config["grad_clip"], batch=config["wake_batch"],
        T_wake=config["wake_T"], input_dim=config["input_dim"],
        collect_snapshots=False, task_seed_fn=lambda i: i, verbose=False)


# ── Per-seed body ────────────────────────────────────────────────────────────

def per_seed(seed: int, device, config: dict) -> dict:
    L = config["stream_length"]
    K_middle = L - 1

    base_freqs = _sample_freqs(K_middle, config["base_freqs_offset"] + seed,
                                config["freq_log_lo"], config["freq_log_hi"])
    alt_freqs  = _sample_freqs(K_middle, config["alt_freqs_offset"] + seed,
                                config["freq_log_lo"], config["freq_log_hi"])
    anchor     = float(_sample_freqs(1, config["anchor_freq_offset"] + seed,
                                      config["freq_log_lo"], config["freq_log_hi"])[0])

    rng_a = np.random.default_rng(config["perm_A_offset"] + seed)
    rng_b = np.random.default_rng(config["perm_B_offset"] + seed)
    perm_A = rng_a.permutation(K_middle)
    perm_B = rng_b.permutation(K_middle)
    # Rare: same permutation as A - reroll
    if np.array_equal(perm_A, perm_B):
        perm_B = np.random.default_rng(config["perm_B_offset"] + seed + 12345).permutation(K_middle)

    anchor_spec = _spec(anchor)
    stream_A = [_spec(base_freqs[i]) for i in perm_A] + [anchor_spec]
    stream_B = [_spec(base_freqs[i]) for i in perm_B] + [anchor_spec]
    stream_T = [_spec(alt_freqs[i])  for i in perm_A] + [anchor_spec]

    def _train_one(stream):
        m = _fresh_leakyrnn(seed, config, device)
        r = _train(m, stream, config, device)
        idle = r["final_idle"]
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return idle

    idle_A = _train_one(stream_A)
    idle_B = _train_one(stream_B)
    idle_T = _train_one(stream_T)

    ang_n2_AB = subspace_angle(idle_A, idle_B, n=2)
    ang_n2_AT = subspace_angle(idle_A, idle_T, n=2)
    ang_n3_AB = subspace_angle(idle_A, idle_B, n=3)
    ang_n3_AT = subspace_angle(idle_A, idle_T, n=3)
    l2_AB = l2_divergence(idle_A, idle_B)
    l2_AT = l2_divergence(idle_A, idle_T)

    def _ratio(a, b): return float(a / b) if b > 0 else float("nan")

    return {
        "anchor_freq":   anchor,
        "base_freqs":    base_freqs.tolist(),
        "alt_freqs":     alt_freqs.tolist(),
        "perm_A":        perm_A.tolist(),
        "perm_B":        perm_B.tolist(),
        "stream_A":      [s.to_dict() for s in stream_A],
        "stream_B":      [s.to_dict() for s in stream_B],
        "stream_T":      [s.to_dict() for s in stream_T],
        "distances": {
            "angle_n2_A_vs_B_middle_order":   float(ang_n2_AB),
            "angle_n2_A_vs_T_middle_content": float(ang_n2_AT),
            "angle_n3_A_vs_B_middle_order":   float(ang_n3_AB),
            "angle_n3_A_vs_T_middle_content": float(ang_n3_AT),
            "l2_A_vs_B_middle_order":         float(l2_AB),
            "l2_A_vs_T_middle_content":       float(l2_AT),
        },
        "ratios_order_over_content": {
            "angle_n2": _ratio(ang_n2_AB, ang_n2_AT),
            "angle_n3": _ratio(ang_n3_AB, ang_n3_AT),
            "l2":       _ratio(l2_AB,     l2_AT),
        },
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(per_seed_results: dict[int, dict], config: dict) -> dict:
    seeds = sorted(per_seed_results.keys())
    n_boot = config["bootstrap_resamples"]
    ci_lvl = config["bootstrap_ci"]
    r = per_seed_results

    def _bs(key, seed_off):
        arr = np.array([r[s]["distances"][key] for s in seeds])
        return bootstrap_ci_mean(arr, n_boot, ci_lvl, seed=seed_off)

    distances = {
        "angle_n2_A_vs_B_middle_order":   _bs("angle_n2_A_vs_B_middle_order",   100),
        "angle_n2_A_vs_T_middle_content": _bs("angle_n2_A_vs_T_middle_content", 101),
        "angle_n3_A_vs_B_middle_order":   _bs("angle_n3_A_vs_B_middle_order",   102),
        "angle_n3_A_vs_T_middle_content": _bs("angle_n3_A_vs_T_middle_content", 103),
        "l2_A_vs_B_middle_order":         _bs("l2_A_vs_B_middle_order",         104),
        "l2_A_vs_T_middle_content":       _bs("l2_A_vs_T_middle_content",       105),
    }

    def _bs_ratio(key, seed_off):
        arr = np.array([r[s]["ratios_order_over_content"][key] for s in seeds])
        ci = bootstrap_ci_mean(arr, n_boot, ci_lvl, seed=seed_off)
        return {
            "bootstrap_ci": ci,
            "per_seed":     arr.tolist(),
            "n_below_one":  int((arr < 1.0).sum()),
        }

    ratios = {
        "angle_n2": _bs_ratio("angle_n2", 200),
        "angle_n3": _bs_ratio("angle_n3", 201),
        "l2":       _bs_ratio("l2",       202),
    }

    return {
        "n_seeds": len(seeds),
        "stream_length": config["stream_length"],
        "K_middle": config["stream_length"] - 1,
        "distances": distances,
        "ratios_order_over_content": ratios,
    }


# ── Entry point ──────────────────────────────────────────────────────────────

def run(device) -> dict:
    torch.set_default_dtype(torch.float32)
    seeds = list(range(CONFIG["n_seeds"]))
    final, _ = run_seeded(
        name=NAME, version=VERSION, seeds=seeds,
        per_seed_fn=per_seed, aggregate_fn=aggregate,
        config=CONFIG, device=device,
        description=DESCRIPTION, details=DETAILS, verbose=True,
    )

    a = final["aggregated"]
    print("\n" + "=" * 60)
    print(f"ORDER_EFFECT_ANCHORED v{VERSION} summary")
    print("=" * 60)
    print(f"n_seeds = {a['n_seeds']}, stream_length = {a['stream_length']} "
          f"(middle K = {a['K_middle']}, plus 1 shared anchor)")
    print()
    print("Absolute distances (both share the same LAST task):")
    for k, v in a["distances"].items():
        print(f"  {k:36s} = {v['mean']:6.2f}  95% CI [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]")
    print()
    print("Ratios middle-order / middle-content:")
    for m in ["angle_n2", "angle_n3", "l2"]:
        r = a["ratios_order_over_content"][m]
        ci = r["bootstrap_ci"]
        print(f"  {m:8s} ratio = {ci['mean']:.3f}  "
              f"95% CI [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]  "
              f"below 1: {r['n_below_one']}/{a['n_seeds']}")
    print()
    print("Interpretation guide:")
    print("  If A_vs_B_middle_order angles are near 0 → middle order does")
    print("  not matter; the entire signature came from recency (last task).")
    print("  If comparable to v2's ~72° → middle order matters as much as")
    print("  full-stream variation, order really is load-bearing.")
    print("  If intermediate → middle order contributes partially.")
    return None
