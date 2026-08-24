"""
Industry-standard replication of the B5 sleep-effect claim.

Paper's B5 claim: adding a Hebbian sleep phase between wake-training
cycles AMPLIFIES the divergence between networks trained on different
streams (+10% L2, reduced subspace angle). Reported at n=1 seed.

This experiment tests whether sleep is a load-bearing component of the
architecture.

Design (per seed):
    Sample two independent streams S_A, S_B of length L.
    Train FOUR networks (all from the SAME torch.manual_seed(seed) init):
        w_A: wake+sleep on S_A
        w_B: wake+sleep on S_B
        n_A: wake-only  on S_A
        n_B: wake-only  on S_B

    with_sleep_distance    = distance(w_A, w_B)
    without_sleep_distance = distance(n_A, n_B)
    delta = with_sleep_distance - without_sleep_distance

Bootstrap CI on the paired delta over seeds.

Verdict:
    - CI > 0: sleep AMPLIFIES divergence (paper's claim).
    - CI < 0: sleep REDUCES divergence.
    - CI includes 0: sleep has no measurable effect on divergence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.harness import run_seeded
from src.metrics import (bootstrap_ci_mean, paired_delta_ci,
                         subspace_angle, l2_divergence)
from src.model import LeakyRNN
from src.tasks import sample_specs
from src.train import train_stream


NAME = "sleep_effect_hardened"
DESCRIPTION = ("Industry-standard replication of the B5 sleep-effect claim. "
               "n=30 seeds. Four networks per seed (2 with-sleep, 2 "
               "without-sleep) on two independent streams. Bootstrap CI "
               "on paired (with_sleep - without_sleep) divergence delta.")
DETAILS = ("v1: LeakyRNN (hidden_dim=256, τ=5, target_sr=0.95, α=0.8). "
           "Stream length L=5, per-seed sinusoid tasks with log-uniform "
           "frequency [0.5, 20] Hz. Wake_phase 200 steps, sleep_phase "
           "600 steps, η=0.01, decay=0.001, ACh gate 0.3. Idle 400 "
           "steps. Subspace angle at n=2 and n=3, L2 divergence. "
           "Bootstrap 2000 resamples for 95% CI on each paired delta.")
VERSION = "v1"


CONFIG: dict[str, Any] = {
    "hidden_dim": 256,
    "input_dim":  16,
    "output_dim": 16,
    "tau":        5.0,
    "target_sr":  0.95,
    "alpha":      0.8,
    "task_family": "sine",
    "stream_length": 5,
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
    "stream_A_seed_offset": 100000,
    "stream_B_seed_offset": 200000,
}


# ── Per-seed body ────────────────────────────────────────────────────────────

def _fresh_leakyrnn(seed: int, config: dict, device):
    torch.manual_seed(seed)
    return LeakyRNN(config["input_dim"], config["hidden_dim"],
                    config["output_dim"], tau=config["tau"],
                    alpha=config["alpha"],
                    target_sr=config["target_sr"]).to(device)


def _train(model, stream, with_sleep, config, device):
    return train_stream(
        model, stream, device, with_sleep=with_sleep,
        wake_steps=config["wake_steps"], sleep_steps=config["sleep_steps"],
        idle_steps=config["idle_steps"], lr=config["wake_lr"],
        eta=config["eta"], decay=config["decay"],
        wake_ach=config["wake_ach"], sleep_ach=config["sleep_ach"],
        grad_clip=config["grad_clip"], batch=config["wake_batch"],
        T_wake=config["wake_T"], input_dim=config["input_dim"],
        collect_snapshots=False, task_seed_fn=lambda i: i, verbose=False)


def per_seed(seed: int, device, config: dict) -> dict:
    L = config["stream_length"]
    stream_A = sample_specs(config["task_family"], L,
                             seed=config["stream_A_seed_offset"] + seed)
    stream_B = sample_specs(config["task_family"], L,
                             seed=config["stream_B_seed_offset"] + seed)

    def _train_one(stream, with_sleep):
        m = _fresh_leakyrnn(seed, config, device)
        r = _train(m, stream, with_sleep, config, device)
        idle = r["final_idle"]
        diag = r["sleep_diagnostics"]
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return idle, diag

    idle_w_A, diag_w_A = _train_one(stream_A, with_sleep=True)
    idle_w_B, diag_w_B = _train_one(stream_B, with_sleep=True)
    idle_n_A, diag_n_A = _train_one(stream_A, with_sleep=False)
    idle_n_B, diag_n_B = _train_one(stream_B, with_sleep=False)

    return {
        "stream_A": [s.to_dict() for s in stream_A],
        "stream_B": [s.to_dict() for s in stream_B],
        "with_sleep": {
            "angle_n2": subspace_angle(idle_w_A, idle_w_B, n=2),
            "angle_n3": subspace_angle(idle_w_A, idle_w_B, n=3),
            "l2":       l2_divergence(idle_w_A, idle_w_B),
        },
        "without_sleep": {
            "angle_n2": subspace_angle(idle_n_A, idle_n_B, n=2),
            "angle_n3": subspace_angle(idle_n_A, idle_n_B, n=3),
            "l2":       l2_divergence(idle_n_A, idle_n_B),
        },
        "sleep_diagnostics": {
            "w_A_final_sr": float(diag_w_A[-1]["post_sr"]),
            "w_B_final_sr": float(diag_w_B[-1]["post_sr"]),
            "w_A_any_reverted": bool(any(d.get("reverted", False) for d in diag_w_A)),
            "w_B_any_reverted": bool(any(d.get("reverted", False) for d in diag_w_B)),
        },
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def _verdict(ci: dict, label: str) -> str:
    if ci["ci_lo"] > 0.0:
        return f"{label}: SLEEP AMPLIFIES (CI > 0)"
    if ci["ci_hi"] < 0.0:
        return f"{label}: SLEEP REDUCES (CI < 0)"
    return f"{label}: INCONCLUSIVE (CI crosses 0)"


def aggregate(per_seed_results: dict[int, dict], config: dict) -> dict:
    seeds = sorted(per_seed_results.keys())
    n_boot = config["bootstrap_resamples"]
    ci_lvl = config["bootstrap_ci"]

    r = per_seed_results
    out: dict = {"n_seeds": len(seeds),
                 "stream_length": config["stream_length"]}

    for metric in ["angle_n2", "angle_n3", "l2"]:
        w = np.array([r[s]["with_sleep"][metric] for s in seeds])
        n = np.array([r[s]["without_sleep"][metric] for s in seeds])
        delta = w - n
        ci_w = bootstrap_ci_mean(w, n_boot, ci_lvl, seed=100)
        ci_n = bootstrap_ci_mean(n, n_boot, ci_lvl, seed=101)
        ci_d = paired_delta_ci(w, n, n_boot, ci_lvl, seed=102)
        out[metric] = {
            "with_sleep_mean":    ci_w,
            "without_sleep_mean": ci_n,
            "paired_delta":       ci_d,
            "per_seed_with":      w.tolist(),
            "per_seed_without":   n.tolist(),
        }

    out["verdicts"] = [
        _verdict(out["angle_n2"]["paired_delta"], "delta_angle_n2"),
        _verdict(out["angle_n3"]["paired_delta"], "delta_angle_n3"),
        _verdict(out["l2"]["paired_delta"],       "delta_l2      "),
    ]
    return out


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
    print(f"SLEEP_EFFECT_HARDENED v{VERSION} summary")
    print("=" * 60)
    print(f"n_seeds = {a['n_seeds']}, stream_length = {a['stream_length']}")
    for metric in ["angle_n2", "angle_n3", "l2"]:
        w = a[metric]["with_sleep_mean"]
        n = a[metric]["without_sleep_mean"]
        d = a[metric]["paired_delta"]
        print(f"\n  {metric}:")
        print(f"    with_sleep    = {w['mean']:.3f}  95% CI [{w['ci_lo']:.3f}, {w['ci_hi']:.3f}]")
        print(f"    without_sleep = {n['mean']:.3f}  95% CI [{n['ci_lo']:.3f}, {n['ci_hi']:.3f}]")
        print(f"    paired delta  = {d['mean']:+.3f}  95% CI [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]")
        print(f"    seeds where with > without: {d['n_seeds_a_greater']}/{a['n_seeds']}")
    print("\n  Verdicts:")
    for v in a["verdicts"]:
        print(f"    - {v}")
    return None
