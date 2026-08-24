"""
Path-dependence hardened: subspace angle vs stream length, n=30 seeds.

Paper's B4 claim: divergence between two networks trained on independent
streams grows with stream length. Reported at n=1 for lengths
{1, 2, 3, 5, 7, 10}.

This experiment measures the same relationship at n=30 seeds with
bootstrap CIs at each length.

Design (per seed):

    Sample two INDEPENDENT random streams, each of length L_max = 10.
    Train N_A on stream_A for L_max wake-sleep cycles; snapshot idle
    trajectory after EVERY cycle. Do the same for N_B on stream_B.

    For each length L in {1, 2, 3, 5, 7, 10}, compute
        angle_L = subspace_angle(idle_A_after_cycle_L,
                                 idle_B_after_cycle_L,
                                 n=2)
    plus the same at n=3 and the L2 divergence.

    30 seeds → per-length bootstrap 95% CI on the mean angle.

Verdict per metric:
    Slope of angle vs log(L): positive with CI excluding zero means
    divergence grows monotonically with stream length (B4 claim).
    Flat or negative means B4 does not hold.

This is more efficient than training a fresh pair of networks per length
(the L=10 training subsumes L=5, L=3, etc. through per-cycle snapshots).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.harness import run_seeded
from src.metrics import (bootstrap_ci_mean, bootstrap_ci_slope,
                         subspace_angle, l2_divergence)
from src.model import LeakyRNN
from src.tasks import sample_specs
from src.train import train_stream


NAME = "path_dependence_hardened"
DESCRIPTION = ("Divergence vs stream length at n=30 seeds. Two "
               "independent random streams per seed, 10 wake-sleep "
               "cycles each, snapshots at every cycle. Bootstrap CI on "
               "mean angle and L2 at each length, plus a slope test.")
DETAILS = ("v1: LeakyRNN (hidden_dim=256, τ=5, target_sr=0.95, α=0.8). "
           "Streams sampled per-seed from sinusoid task family, K=10 "
           "log-uniform freqs. Wake+sleep training, canonical config. "
           "Idle 400 steps. Lengths measured: 1, 2, 3, 5, 7, 10 cycles. "
           "Bootstrap 2000 resamples, 95% CI at each length; slope of "
           "angle vs log(length) also reported with bootstrap CI.")
VERSION = "v1"


CONFIG: dict[str, Any] = {
    "hidden_dim": 256,
    "input_dim":  16,
    "output_dim": 16,
    "tau":        5.0,
    "target_sr":  0.95,
    "alpha":      0.8,
    "task_family":  "sine",
    "stream_length_max": 10,
    "lengths_measured": [1, 2, 3, 5, 7, 10],
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
    "stream_A_seed_offset": 1300000,
    "stream_B_seed_offset": 1400000,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fresh_leakyrnn(seed: int, config: dict, device):
    torch.manual_seed(seed)
    return LeakyRNN(config["input_dim"], config["hidden_dim"],
                    config["output_dim"], tau=config["tau"],
                    alpha=config["alpha"],
                    target_sr=config["target_sr"]).to(device)


def _train_full(model, stream, config, device):
    """Train through the full stream, taking a snapshot after each cycle."""
    return train_stream(
        model, stream, device, with_sleep=True,
        wake_steps=config["wake_steps"], sleep_steps=config["sleep_steps"],
        idle_steps=config["idle_steps"], lr=config["wake_lr"],
        eta=config["eta"], decay=config["decay"],
        wake_ach=config["wake_ach"], sleep_ach=config["sleep_ach"],
        grad_clip=config["grad_clip"], batch=config["wake_batch"],
        T_wake=config["wake_T"], input_dim=config["input_dim"],
        collect_snapshots=True, task_seed_fn=lambda i: i, verbose=False)


# ── Per-seed body ────────────────────────────────────────────────────────────

def per_seed(seed: int, device, config: dict) -> dict:
    L_max = config["stream_length_max"]
    stream_A = sample_specs(config["task_family"], L_max,
                            seed=config["stream_A_seed_offset"] + seed)
    stream_B = sample_specs(config["task_family"], L_max,
                            seed=config["stream_B_seed_offset"] + seed)

    m_A = _fresh_leakyrnn(seed, config, device)
    r_A = _train_full(m_A, stream_A, config, device)
    snaps_A = r_A["snapshots"]
    del m_A
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    m_B = _fresh_leakyrnn(seed, config, device)
    r_B = _train_full(m_B, stream_B, config, device)
    snaps_B = r_B["snapshots"]
    del m_B
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if len(snaps_A) != L_max or len(snaps_B) != L_max:
        raise RuntimeError(
            f"expected {L_max} snapshots, got A={len(snaps_A)}, B={len(snaps_B)}")

    per_length: dict[int, dict] = {}
    for L in config["lengths_measured"]:
        sa = snaps_A[L - 1]      # snapshot AFTER cycle L (0-indexed → L-1)
        sb = snaps_B[L - 1]
        per_length[L] = {
            "angle_n2": subspace_angle(sa, sb, n=2),
            "angle_n3": subspace_angle(sa, sb, n=3),
            "l2":       l2_divergence(sa, sb),
        }

    return {
        "stream_A": [s.to_dict() for s in stream_A],
        "stream_B": [s.to_dict() for s in stream_B],
        "per_length": {str(L): v for L, v in per_length.items()},
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(per_seed_results: dict[int, dict], config: dict) -> dict:
    seeds = sorted(per_seed_results.keys())
    lengths = config["lengths_measured"]
    n_boot = config["bootstrap_resamples"]
    ci_lvl = config["bootstrap_ci"]
    r = per_seed_results

    # per-length CIs
    def _length_ci(metric: str, seed_off: int) -> dict:
        out = {}
        for L in lengths:
            arr = np.array([r[s]["per_length"][str(L)][metric] for s in seeds])
            out[str(L)] = bootstrap_ci_mean(arr, n_boot, ci_lvl, seed=seed_off + L)
        return out

    per_length_ci = {
        "angle_n2": _length_ci("angle_n2", 100),
        "angle_n3": _length_ci("angle_n3", 200),
        "l2":       _length_ci("l2",       300),
    }

    # Slope of angle vs log(length), bootstrapped across seeds
    log_L = np.log(np.array(lengths, dtype=float))
    def _slope_ci(metric: str, seed_off: int) -> dict:
        curves = np.stack([[r[s]["per_length"][str(L)][metric] for L in lengths]
                           for s in seeds])  # (n_seeds, n_lengths)
        return bootstrap_ci_slope(log_L, curves, n_boot, ci_lvl, seed=seed_off)

    slope = {
        "angle_n2_vs_logL": _slope_ci("angle_n2", 400),
        "angle_n3_vs_logL": _slope_ci("angle_n3", 401),
        "l2_vs_logL":       _slope_ci("l2",       402),
    }

    def _verdict(ci: dict, label: str) -> str:
        if ci["ci_lo"] > 0:
            return f"{label}: POSITIVE (CI excludes 0)"
        if ci["ci_hi"] < 0:
            return f"{label}: NEGATIVE (CI excludes 0)"
        return f"{label}: INCONCLUSIVE (CI crosses 0)"

    verdicts = [
        _verdict(slope["angle_n2_vs_logL"], "slope angle_n2 vs log(L)"),
        _verdict(slope["angle_n3_vs_logL"], "slope angle_n3 vs log(L)"),
        _verdict(slope["l2_vs_logL"],       "slope l2       vs log(L)"),
    ]

    return {
        "n_seeds": len(seeds),
        "lengths_measured": lengths,
        "per_length_ci": per_length_ci,
        "slope_vs_logL": slope,
        "verdicts": verdicts,
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
    print(f"PATH_DEPENDENCE_HARDENED v{VERSION} summary")
    print("=" * 60)
    print(f"n_seeds = {a['n_seeds']}, lengths = {a['lengths_measured']}")
    print()
    for metric in ["angle_n2", "angle_n3", "l2"]:
        print(f"  {metric} at each length:")
        for L in a["lengths_measured"]:
            ci = a["per_length_ci"][metric][str(L)]
            print(f"    L={L:2d}: {ci['mean']:6.2f}  95% CI [{ci['ci_lo']:.2f}, {ci['ci_hi']:.2f}]")
        s = a["slope_vs_logL"][f"{metric}_vs_logL"]
        print(f"    slope vs log(L): {s['slope_point']:+.3f}  95% CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]")
        print()
    print("Verdicts:")
    for v in a["verdicts"]:
        print(f"  - {v}")
    return None
