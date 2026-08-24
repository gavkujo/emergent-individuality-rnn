"""
Industry-standard replication of the B6 order-effect claim.

Paper's B6 claim: reversing the order of the same sequence of tasks
produces LESS divergence from the forward-order network than training on
a stream of unrelated tasks. Reported ratio (angle(fwd, reversed) /
angle(fwd, unrelated)) ≈ 0.86 at n=1 seed.

This experiment tests whether the ORDER of experience carries a
measurable, seed-robust signature — the paper's key structural claim.

Design (per seed):
    S     — random stream of L tasks
    S_rev — reverse(S), same tasks in the opposite order
    T     — independent random stream of L tasks

    All three networks start from the SAME torch.manual_seed(seed) init.
    Each is trained wake+sleep on its stream.
    Idle trajectories → subspace angle and L2 divergence.

Effect measured (three metrics):
    ratio_angle_n2 = angle_n2(idle_S, idle_Srev) / angle_n2(idle_S, idle_T)
    ratio_angle_n3 = angle_n3(idle_S, idle_Srev) / angle_n3(idle_S, idle_T)
    ratio_l2       = l2(idle_S, idle_Srev)       / l2(idle_S, idle_T)

Verdict per metric:
    - CI < 1 with 95% CI excluding 1: ORDER EFFECT PRESENT (order matters,
      less than content but nonzero).
    - CI includes 1: ORDER EFFECT INCONCLUSIVE (order matters as much as
      or as little as content variation).
    - CI > 1: reverse-order MORE divergent than unrelated content
      (contradicts paper).

Stream length L is a knob. L=5 for direct comparison to the paper.
L=30 planned as a follow-up scale-up (Phase B extension).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.harness import run_seeded
from src.metrics import (bootstrap_ci_mean, subspace_angle, l2_divergence)
from src.model import LeakyRNN
from src.tasks import sample_specs
from src.train import train_stream


NAME = "order_effect_hardened"
DESCRIPTION = ("Industry-standard replication of the B6 order-effect claim. "
               "n=30 seeds, wake+sleep training. Per seed: forward stream, "
               "reversed stream, unrelated content stream; measure ratio "
               "angle(fwd, rev) / angle(fwd, unrelated). Bootstrap CI on "
               "the ratio over seeds.")
DETAILS = ("v1: LeakyRNN (hidden_dim=256, τ=5, target_sr=0.95, α=0.8). "
           "Stream length L=5, sinusoid tasks with log-uniform frequency "
           "on [0.5, 20] Hz. Streams are sampled per-seed with a distinct "
           "sub-seed (order_seed = 100000 + seed for forward, 200000 + "
           "seed for the unrelated). Wake_phase 200 steps, sleep_phase "
           "600 steps, η=0.01, decay=0.001, ACh gate 0.3. Idle 400 steps. "
           "Subspace angle at n=2 (post-hardening default) and n=3 "
           "(legacy). Bootstrap 2000 resamples for 95% CI on each ratio.")
VERSION = "v1"


CONFIG: dict[str, Any] = {
    # Architecture
    "hidden_dim": 256,
    "input_dim":  16,
    "output_dim": 16,
    "tau":        5.0,
    "target_sr":  0.95,
    "alpha":      0.8,
    # Streams
    "task_family":  "sine",
    "stream_length": 5,
    # Wake / sleep
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
    # Statistics
    "bootstrap_resamples": 2000,
    "bootstrap_ci":        0.95,
    # Seeds
    "n_seeds":     30,
    # Stream-seed offsets to keep forward/unrelated streams distinct per seed.
    "forward_stream_seed_offset":   100000,
    "unrelated_stream_seed_offset": 200000,
}


# ── Per-seed body ────────────────────────────────────────────────────────────

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


def per_seed(seed: int, device, config: dict) -> dict:
    # Sample two independent streams for this seed
    L = config["stream_length"]
    stream_fwd = sample_specs(
        config["task_family"], L,
        seed=config["forward_stream_seed_offset"] + seed)
    stream_rev = list(reversed(stream_fwd))
    stream_unrelated = sample_specs(
        config["task_family"], L,
        seed=config["unrelated_stream_seed_offset"] + seed)

    # Train three networks, all from the SAME init (same torch seed)
    m_fwd = _fresh_leakyrnn(seed, config, device)
    r_fwd = _train(m_fwd, stream_fwd, config, device)
    idle_fwd = r_fwd["final_idle"]
    fwd_diag = r_fwd["sleep_diagnostics"]
    del m_fwd
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    m_rev = _fresh_leakyrnn(seed, config, device)
    r_rev = _train(m_rev, stream_rev, config, device)
    idle_rev = r_rev["final_idle"]
    rev_diag = r_rev["sleep_diagnostics"]
    del m_rev
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    m_unr = _fresh_leakyrnn(seed, config, device)
    r_unr = _train(m_unr, stream_unrelated, config, device)
    idle_unr = r_unr["final_idle"]
    unr_diag = r_unr["sleep_diagnostics"]
    del m_unr
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Divergence metrics
    angle_n2_rev = subspace_angle(idle_fwd, idle_rev, n=2)
    angle_n2_unr = subspace_angle(idle_fwd, idle_unr, n=2)
    angle_n3_rev = subspace_angle(idle_fwd, idle_rev, n=3)
    angle_n3_unr = subspace_angle(idle_fwd, idle_unr, n=3)
    l2_rev = l2_divergence(idle_fwd, idle_rev)
    l2_unr = l2_divergence(idle_fwd, idle_unr)

    def _ratio(a, b): return float(a / b) if b > 0 else float("nan")

    return {
        "stream_fwd":       [s.to_dict() for s in stream_fwd],
        "stream_unrelated": [s.to_dict() for s in stream_unrelated],
        "distances": {
            "angle_n2_fwd_vs_rev": float(angle_n2_rev),
            "angle_n2_fwd_vs_unr": float(angle_n2_unr),
            "angle_n3_fwd_vs_rev": float(angle_n3_rev),
            "angle_n3_fwd_vs_unr": float(angle_n3_unr),
            "l2_fwd_vs_rev":       float(l2_rev),
            "l2_fwd_vs_unr":       float(l2_unr),
        },
        "ratios": {
            "angle_n2": _ratio(angle_n2_rev, angle_n2_unr),
            "angle_n3": _ratio(angle_n3_rev, angle_n3_unr),
            "l2":       _ratio(l2_rev, l2_unr),
        },
        "sleep_diagnostics": {
            "fwd_final_sr": float(fwd_diag[-1]["post_sr"]),
            "rev_final_sr": float(rev_diag[-1]["post_sr"]),
            "unr_final_sr": float(unr_diag[-1]["post_sr"]),
            "fwd_any_reverted": bool(any(d.get("reverted", False) for d in fwd_diag)),
            "rev_any_reverted": bool(any(d.get("reverted", False) for d in rev_diag)),
            "unr_any_reverted": bool(any(d.get("reverted", False) for d in unr_diag)),
        },
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def _verdict(ci: dict, null_value: float, label: str) -> str:
    if ci["ci_hi"] < null_value:
        return f"{label}: BELOW {null_value:.3f} (order effect present)"
    if ci["ci_lo"] > null_value:
        return f"{label}: ABOVE {null_value:.3f} (reverse MORE divergent than unrelated)"
    return f"{label}: INCONCLUSIVE (CI crosses {null_value:.3f})"


def aggregate(per_seed_results: dict[int, dict], config: dict) -> dict:
    seeds = sorted(per_seed_results.keys())
    n_boot = config["bootstrap_resamples"]
    ci_lvl = config["bootstrap_ci"]

    # Assemble per-seed arrays
    r = per_seed_results
    d_n2_rev = np.array([r[s]["distances"]["angle_n2_fwd_vs_rev"] for s in seeds])
    d_n2_unr = np.array([r[s]["distances"]["angle_n2_fwd_vs_unr"] for s in seeds])
    d_n3_rev = np.array([r[s]["distances"]["angle_n3_fwd_vs_rev"] for s in seeds])
    d_n3_unr = np.array([r[s]["distances"]["angle_n3_fwd_vs_unr"] for s in seeds])
    d_l2_rev = np.array([r[s]["distances"]["l2_fwd_vs_rev"] for s in seeds])
    d_l2_unr = np.array([r[s]["distances"]["l2_fwd_vs_unr"] for s in seeds])

    ratio_n2 = np.array([r[s]["ratios"]["angle_n2"] for s in seeds])
    ratio_n3 = np.array([r[s]["ratios"]["angle_n3"] for s in seeds])
    ratio_l2 = np.array([r[s]["ratios"]["l2"] for s in seeds])

    ci_n2 = bootstrap_ci_mean(ratio_n2, n_boot, ci_lvl, seed=0)
    ci_n3 = bootstrap_ci_mean(ratio_n3, n_boot, ci_lvl, seed=1)
    ci_l2 = bootstrap_ci_mean(ratio_l2, n_boot, ci_lvl, seed=2)

    # Sign counts: how many seeds show reverse-order < unrelated-content
    n_below_one_n2 = int(np.sum(ratio_n2 < 1.0))
    n_below_one_n3 = int(np.sum(ratio_n3 < 1.0))
    n_below_one_l2 = int(np.sum(ratio_l2 < 1.0))

    return {
        "n_seeds": len(seeds),
        "stream_length": config["stream_length"],
        "distances": {
            "angle_n2_fwd_vs_rev": bootstrap_ci_mean(d_n2_rev, n_boot, ci_lvl, seed=10),
            "angle_n2_fwd_vs_unr": bootstrap_ci_mean(d_n2_unr, n_boot, ci_lvl, seed=11),
            "angle_n3_fwd_vs_rev": bootstrap_ci_mean(d_n3_rev, n_boot, ci_lvl, seed=12),
            "angle_n3_fwd_vs_unr": bootstrap_ci_mean(d_n3_unr, n_boot, ci_lvl, seed=13),
            "l2_fwd_vs_rev":       bootstrap_ci_mean(d_l2_rev, n_boot, ci_lvl, seed=14),
            "l2_fwd_vs_unr":       bootstrap_ci_mean(d_l2_unr, n_boot, ci_lvl, seed=15),
        },
        "ratios": {
            "angle_n2": {
                "bootstrap_ci": ci_n2,
                "per_seed":     ratio_n2.tolist(),
                "n_below_one":  n_below_one_n2,
            },
            "angle_n3": {
                "bootstrap_ci": ci_n3,
                "per_seed":     ratio_n3.tolist(),
                "n_below_one":  n_below_one_n3,
            },
            "l2": {
                "bootstrap_ci": ci_l2,
                "per_seed":     ratio_l2.tolist(),
                "n_below_one":  n_below_one_l2,
            },
        },
        "verdicts": [
            _verdict(ci_n2, 1.0, "ratio_angle_n2 vs 1"),
            _verdict(ci_n3, 1.0, "ratio_angle_n3 vs 1"),
            _verdict(ci_l2, 1.0, "ratio_l2       vs 1"),
        ],
    }


# ── Entry point ──────────────────────────────────────────────────────────────

def run(device) -> dict:
    torch.set_default_dtype(torch.float32)
    seeds = list(range(CONFIG["n_seeds"]))
    final, _ = run_seeded(
        name=NAME,
        version=VERSION,
        seeds=seeds,
        per_seed_fn=per_seed,
        aggregate_fn=aggregate,
        config=CONFIG,
        device=device,
        description=DESCRIPTION,
        details=DETAILS,
        verbose=True,
    )

    a = final["aggregated"]
    print("\n" + "=" * 60)
    print(f"ORDER_EFFECT_HARDENED v{VERSION} summary")
    print("=" * 60)
    print(f"n_seeds = {a['n_seeds']}, stream_length = {a['stream_length']}")
    print("\n  Absolute distances (means over seeds):")
    for k, v in a["distances"].items():
        print(f"    {k:22s} = {v['mean']:.3f}  95% CI [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
    print("\n  Ratios (fwd_vs_rev / fwd_vs_unrelated):")
    for metric in ["angle_n2", "angle_n3", "l2"]:
        r = a["ratios"][metric]
        ci = r["bootstrap_ci"]
        print(f"    {metric:8s}  ratio = {ci['mean']:.3f}  "
              f"95% CI [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]  "
              f"below 1: {r['n_below_one']}/{a['n_seeds']}")
    print("\n  Verdicts:")
    for v in a["verdicts"]:
        print(f"    - {v}")
    return None
