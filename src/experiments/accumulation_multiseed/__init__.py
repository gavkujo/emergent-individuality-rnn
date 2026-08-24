"""
Multi-seed test of the B3 divergence-accumulation claim.

The workshop paper's B3 claim is: "subspace angle between two networks
trained on different experiential streams grows across wake-sleep cycles."
The original benchmark reported this at n=1 seed. Three-seed testing in
`subspace_dim` showed non-robust behaviour; here we replicate at n_seeds =
10, at both n=2 (principled after subspace_dim) and n=3 (legacy), for both
metrics the paper reports (subspace angle AND L2 divergence).

The output settles whether the accumulation claim survives multi-seed
verification. Two possible outcomes:

  A. Slope is significantly positive at 10 seeds, with a CI that
     excludes zero. Paper claim survives with proper error bars.
  B. Slope is not distinguishable from zero. Paper claim must be dropped
     or reframed (see contingency in todo_hardening.md).

Streams: A_low vs B_high (same as B3 in benchmark). Metrics per cycle:
subspace_angle at n=2 and n=3, L2 divergence (mean of ‖s_a(t) − s_b(t)‖).
Slope regressed vs cycle index with a bootstrap 95% CI over seeds.
"""

from __future__ import annotations

import numpy as np
import torch

from src.model import LeakyRNN
from src.train import (make_sine_task, wake_phase, sleep_phase,
                       run_idle)


NAME = "accumulation_multiseed"
DESCRIPTION = ("Multi-seed test of the B3 accumulation claim. 10 seeds, "
               "A_low vs B_high, cycle-by-cycle subspace angle and L2 "
               "divergence, with bootstrap CI on the slope.")
DETAILS = ("v1: LeakyRNN (seed ∈ {42..51}, tau=5, target_sr=0.95, α=0.8, "
           "s_in=s_out=0.1). Wake_phase 200 steps, sleep_phase 600 steps "
           "with η=0.01, decay=0.001, ACh gate 0.3. Idle 300 steps per "
           "cycle. Subspace angle computed at n=2 (post-hardening default) "
           "and n=3 (legacy). Bootstrap resamples over seeds: 2000. "
           "This is the direct multi-seed replication of the B3 benchmark.")
VERSION = "v1"


# Config — mirrors the current benchmark.py exactly, except for the seed set.
INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 16, 128, 16
TAU, SR_INIT, ALPHA = 5.0, 0.95, 0.8
ETA, DECAY = 0.01, 0.001
IDLE_STEPS = 300
WAKE_STEPS = 200
SLEEP_STEPS = 600

STREAM_A = [1.0, 1.5, 2.0, 1.0, 2.0]
STREAM_B = [8.0, 12.0, 6.0, 8.0, 6.0]

SEEDS = list(range(42, 52))                  # 10 seeds
N_VALUES = [2, 3]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CI = 0.95


# ── Metrics ──────────────────────────────────────────────────────────────────

def _subspace_angle_at_n(sa: np.ndarray, sb: np.ndarray, n: int) -> float:
    a = sa - sa.mean(0)
    b = sb - sb.mean(0)
    _, _, Va = np.linalg.svd(a, full_matrices=False)
    _, _, Vb = np.linalg.svd(b, full_matrices=False)
    sv = np.clip(np.linalg.svd(Va[:n] @ Vb[:n].T, compute_uv=False), -1, 1)
    return float(np.arccos(sv).mean() * 180 / np.pi)


def _l2_divergence(sa: np.ndarray, sb: np.ndarray) -> float:
    """Mean over time of ‖s_a(t) − s_b(t)‖."""
    return float(np.linalg.norm(sa - sb, axis=-1).mean())


# ── Training a stream ────────────────────────────────────────────────────────

def _train_stream_return_snaps(seed: int, stream: list[float], device) -> list[np.ndarray]:
    """Train a fresh LeakyRNN(seed=seed) on `stream` with wake+sleep.
    Return list of per-cycle idle snapshots (one per wake-sleep cycle)."""
    torch.manual_seed(seed)
    m = LeakyRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
                 tau=TAU, alpha=ALPHA, target_sr=SR_INIT).to(device)
    snaps: list[np.ndarray] = []
    for i, freq in enumerate(stream):
        wake_phase(m, *make_sine_task(freq, seed=i, device=device),
                   steps=WAKE_STEPS, lr=3e-3, ach_gate=1.0)
        sleep_phase(m, sleep_steps=SLEEP_STEPS, eta=ETA, decay=DECAY,
                    ach_gate=0.3, device=device)
        snaps.append(run_idle(m, steps=IDLE_STEPS, device=device))
    return snaps


# ── Slope regression + bootstrap ─────────────────────────────────────────────

def _slope(cycles: np.ndarray, values: np.ndarray) -> float:
    """OLS slope of `values` vs `cycles`."""
    return float(np.polyfit(cycles, values, 1)[0])


def _bootstrap_slope_ci(per_seed_curves: np.ndarray, cycles: np.ndarray,
                        n_resamples: int, ci_level: float,
                        rng: np.random.Generator) -> tuple[float, float, float]:
    """
    per_seed_curves: (n_seeds, n_cycles) matrix of metric values.
    Fit slope per bootstrap resample of seeds; return (median_slope,
    lo_percentile, hi_percentile).
    """
    n_seeds = per_seed_curves.shape[0]
    slopes = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        resample = per_seed_curves[idx].mean(0)
        slopes[r] = _slope(cycles, resample)
    lo = 0.5 - ci_level / 2.0
    hi = 0.5 + ci_level / 2.0
    return (float(np.median(slopes)),
            float(np.quantile(slopes, lo)),
            float(np.quantile(slopes, hi)))


# ── Main experiment ──────────────────────────────────────────────────────────

def run(device) -> dict:
    torch.set_default_dtype(torch.float32)
    print(f"\naccumulation_multiseed: seeds={SEEDS}, streams=A_low vs B_high, "
          f"device={device}")

    n_cycles = len(STREAM_A)
    cycles = np.arange(1, n_cycles + 1)

    # per_seed[metric_key][seed_idx, cycle_idx] = value
    #
    # metric_key ∈ {"angle_n2", "angle_n3", "l2"}
    per_seed: dict[str, np.ndarray] = {
        "angle_n2": np.zeros((len(SEEDS), n_cycles)),
        "angle_n3": np.zeros((len(SEEDS), n_cycles)),
        "l2":       np.zeros((len(SEEDS), n_cycles)),
    }

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  [seed {seed}] training A_low...", flush=True)
        snaps_a = _train_stream_return_snaps(seed, STREAM_A, device)
        print(f"  [seed {seed}] training B_high...", flush=True)
        snaps_b = _train_stream_return_snaps(seed, STREAM_B, device)

        for c_idx, (sa, sb) in enumerate(zip(snaps_a, snaps_b)):
            per_seed["angle_n2"][seed_idx, c_idx] = _subspace_angle_at_n(sa, sb, 2)
            per_seed["angle_n3"][seed_idx, c_idx] = _subspace_angle_at_n(sa, sb, 3)
            per_seed["l2"][seed_idx, c_idx] = _l2_divergence(sa, sb)

        print(f"  [seed {seed}] cycles 1-5 angle_n2:  " +
              "  ".join(f"{v:5.2f}" for v in per_seed["angle_n2"][seed_idx]))
        print(f"                       angle_n3:  " +
              "  ".join(f"{v:5.2f}" for v in per_seed["angle_n3"][seed_idx]))
        print(f"                       l2:        " +
              "  ".join(f"{v:5.3f}" for v in per_seed["l2"][seed_idx]))

    # ── Aggregate stats + bootstrap ─────────────────────────────────────────
    rng = np.random.default_rng(0)
    print(f"\n\nAggregate over {len(SEEDS)} seeds:")

    metric_summaries: dict[str, dict] = {}
    for key, curves in per_seed.items():
        means = curves.mean(axis=0)
        stds = curves.std(axis=0, ddof=1)
        sems = stds / np.sqrt(len(SEEDS))
        delta_5_minus_1 = curves[:, -1] - curves[:, 0]
        per_seed_slope = np.array([_slope(cycles, curves[i]) for i in range(len(SEEDS))])

        # Bootstrap CI on the mean slope
        med_slope, lo, hi = _bootstrap_slope_ci(
            curves, cycles, BOOTSTRAP_RESAMPLES, BOOTSTRAP_CI, rng)

        # Point estimate: slope of the mean curve
        mean_curve_slope = _slope(cycles, means)

        # Sign test
        n_positive = int(np.sum(delta_5_minus_1 > 0))

        summary = {
            "per_cycle_mean":   means.tolist(),
            "per_cycle_std":    stds.tolist(),
            "per_cycle_sem":    sems.tolist(),
            "per_seed_slope":   per_seed_slope.tolist(),
            "per_seed_delta_c5_minus_c1": delta_5_minus_1.tolist(),
            "mean_curve_slope": mean_curve_slope,
            "bootstrap_slope": {
                "median": med_slope,
                "ci_lo":  lo,
                "ci_hi":  hi,
                "ci_level": BOOTSTRAP_CI,
                "resamples": BOOTSTRAP_RESAMPLES,
            },
            "sign_test": {
                "n_seeds_positive_delta": n_positive,
                "n_total": len(SEEDS),
            },
        }
        metric_summaries[key] = summary

        print(f"\n  {key}:")
        print(f"    mean per cycle:        " +
              "  ".join(f"{v:5.2f}" for v in means))
        print(f"    sem  per cycle:        " +
              "  ".join(f"{v:5.2f}" for v in sems))
        print(f"    slope (mean curve):    {mean_curve_slope:+.3f} °/cycle" if key != "l2"
              else f"    slope (mean curve):    {mean_curve_slope:+.4f}")
        print(f"    slope 95% CI:          [{lo:+.3f}, {hi:+.3f}]")
        print(f"    seeds with positive Δ: {n_positive}/{len(SEEDS)}")

    # ── Verdict ────────────────────────────────────────────────────────────
    print(f"\nVerdict on B3 accumulation:")
    for key in ["angle_n2", "angle_n3", "l2"]:
        ci = metric_summaries[key]["bootstrap_slope"]
        verdict = ("CONFIRMED (CI excludes 0, positive)"
                   if ci["ci_lo"] > 0 else
                   "REJECTED (CI excludes 0, negative)"
                   if ci["ci_hi"] < 0 else
                   "INCONCLUSIVE (CI includes 0)")
        print(f"  {key}: {verdict}  CI=[{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]")

    out = {
        "config": {
            "hidden_dim": HIDDEN_DIM, "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM, "tau": TAU, "target_sr": SR_INIT,
            "alpha": ALPHA, "eta": ETA, "decay": DECAY,
            "wake_steps": WAKE_STEPS, "sleep_steps": SLEEP_STEPS,
            "idle_steps": IDLE_STEPS,
            "seeds": SEEDS, "stream_A_low": STREAM_A,
            "stream_B_high": STREAM_B, "n_cycles": n_cycles,
            "device": str(device),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_ci_level": BOOTSTRAP_CI,
        },
        "per_seed_curves": {k: v.tolist() for k, v in per_seed.items()},
        "cycles": cycles.tolist(),
        "metric_summaries": metric_summaries,
    }

    try:
        from src.experiments.accumulation_multiseed import figures as fig_mod
        fig_mod.render(out)
    except Exception as e:
        print(f"\n(skipping figures: {e})")

    return out
