"""
Effective dimensionality of the idle limit cycle, and sensitivity of the
reported subspace-angle metrics to the choice of `n` in
`train.subspace_angle`.

The current codebase uses `subspace_angle(..., n=3)`. This value was picked,
not derived. This experiment answers:

    1. What is the effective dimensionality of a trained idle trajectory?
       We report the smallest n reaching 90%, 95%, 99% of variance across
       multiple seeds and streams.

    2. How sensitive are the load-bearing benchmark numbers (accumulation,
       sleep effect, order effect) to n?

    3. What n is defensible?

The finding is then written up in `findings/week<NN>_<YYYY>/`.
"""

from __future__ import annotations

import numpy as np
import torch

from src.model import LeakyRNN
from src.train import (make_sine_task, wake_phase, sleep_phase,
                       run_idle, subspace_angle)


NAME = "subspace_dim"
DESCRIPTION = ("Effective dim of idle limit cycle and n-sensitivity of the "
               "subspace_angle metric. Justifies the choice of n or "
               "identifies where the current n=3 is misleading.")
DETAILS = ("v1: LeakyRNN (seed, tau=5, target_sr=0.95, alpha=0.8, "
           "s_in=s_out=0.1). Three streams per seed (A_low, B_high, "
           "D_reversed) trained with the standard wake_phase/sleep_phase. "
           "Idle trajectory has 300 steps. Angles computed at n ∈ {1..8}.")
VERSION = "v1"


INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 16, 128, 16
TAU, SR_INIT, ALPHA = 5.0, 0.95, 0.8
ETA, DECAY = 0.01, 0.001
IDLE_STEPS = 300

STREAMS = {
    "A_low":       [1.0, 1.5, 2.0, 1.0, 2.0],
    "B_high":      [8.0, 12.0, 6.0, 8.0, 6.0],
    "D_reversed":  [2.0, 1.5, 1.0, 2.0, 1.0],
}
SEEDS = [42, 43, 44]
N_VALUES = [1, 2, 3, 4, 5, 8]
VARIANCE_THRESHOLDS = [0.90, 0.95, 0.99]


# ── Metric helpers ────────────────────────────────────────────────────────────

def _svd_centered(states: np.ndarray) -> np.ndarray:
    """SVD of centered states. Returns singular values (descending)."""
    centered = states - states.mean(0)
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    return sv


def _effective_dim(sv: np.ndarray, threshold: float) -> int:
    """Smallest n such that cum(sv²) / total ≥ threshold."""
    var = sv ** 2
    cum = np.cumsum(var) / max(float(var.sum()), 1e-30)
    n = int(np.searchsorted(cum, threshold)) + 1
    return min(n, len(sv))


def _angle_at_n(sa: np.ndarray, sb: np.ndarray, n: int) -> float:
    """Same principle as `train.subspace_angle` but with explicit n."""
    a = sa - sa.mean(0)
    b = sb - sb.mean(0)
    _, _, Va = np.linalg.svd(a, full_matrices=False)
    _, _, Vb = np.linalg.svd(b, full_matrices=False)
    sv = np.clip(np.linalg.svd(Va[:n] @ Vb[:n].T, compute_uv=False), -1, 1)
    return float(np.arccos(sv).mean() * 180 / np.pi)


# ── Training a stream ─────────────────────────────────────────────────────────

def _train_stream(seed: int, stream: list[float], device) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Train a fresh LeakyRNN(seed=seed) on `stream` with wake_phase + sleep_phase.
    Return (per-cycle idle snapshots, final idle trajectory).
    """
    torch.manual_seed(seed)
    m = LeakyRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
                 tau=TAU, alpha=ALPHA, target_sr=SR_INIT).to(device)
    snaps = []
    for i, freq in enumerate(stream):
        wake_phase(m, *make_sine_task(freq, seed=i, device=device),
                   steps=200, lr=3e-3, ach_gate=1.0)
        sleep_phase(m, sleep_steps=600, eta=ETA, decay=DECAY,
                    ach_gate=0.3, device=device)
        snaps.append(run_idle(m, steps=IDLE_STEPS, device=device))
    return snaps, snaps[-1]


# ── Main experiment ───────────────────────────────────────────────────────────

def run(device) -> dict:
    torch.set_default_dtype(torch.float32)
    print(f"\nsubspace_dim: seeds={SEEDS}, streams={list(STREAMS)}, "
          f"device={device}")

    # ── (1) Effective dim across seed × stream ─────────────────────────────
    print(f"\n[1/3] Training + effective-dim measurement...")
    trajectories: dict[tuple[int, str], np.ndarray] = {}
    accum_snapshots: dict[tuple[int, str], list[np.ndarray]] = {}
    effective_dim_rows: list[dict] = []

    for seed in SEEDS:
        for name, stream in STREAMS.items():
            print(f"  training seed={seed}, stream={name}...", flush=True)
            snaps, final = _train_stream(seed, stream, device)
            trajectories[(seed, name)] = final
            accum_snapshots[(seed, name)] = snaps
            sv = _svd_centered(final)
            row = {
                "seed": seed, "stream": name,
                "singular_values_top10": sv[:10].tolist(),
                "variance_top10_pct":
                    (sv[:10] ** 2 / (sv ** 2).sum() * 100).tolist(),
            }
            for t in VARIANCE_THRESHOLDS:
                row[f"n_var_{int(t*100)}"] = _effective_dim(sv, t)
            effective_dim_rows.append(row)
            print(f"    n_var90={row['n_var_90']}, "
                  f"n_var95={row['n_var_95']}, "
                  f"n_var99={row['n_var_99']}, "
                  f"top-3 var%={[round(v,1) for v in row['variance_top10_pct'][:3]]}")

    # ── Distribution summary ───────────────────────────────────────────────
    summary = {}
    for t in VARIANCE_THRESHOLDS:
        vals = [row[f"n_var_{int(t*100)}"] for row in effective_dim_rows]
        summary[f"n_var_{int(t*100)}"] = {
            "min": int(min(vals)),
            "median": int(np.median(vals)),
            "max": int(max(vals)),
            "mean": float(np.mean(vals)),
        }
    print(f"\n  effective-dim summary across {len(SEEDS)*len(STREAMS)} configs:")
    for t in VARIANCE_THRESHOLDS:
        s = summary[f"n_var_{int(t*100)}"]
        print(f"    n_var_{int(t*100)}: min={s['min']}, median={s['median']}, "
              f"max={s['max']}, mean={s['mean']:.2f}")

    # ── (2) n-sensitivity of pairwise angles ───────────────────────────────
    print(f"\n[2/3] Pairwise angle sensitivity to n...")
    #
    # For each seed, compute two angle types:
    #   - "different content"  = angle(A_low, B_high)
    #   - "same content reversed" = angle(A_low, D_reversed)
    # Both at every n ∈ N_VALUES. Report:
    #   - the two angles at each n
    #   - the ratio "same-reversed / different" — the B6 order-effect fraction
    #
    per_seed_pairs: list[dict] = []
    for seed in SEEDS:
        sa = trajectories[(seed, "A_low")]
        sb = trajectories[(seed, "B_high")]
        sd = trajectories[(seed, "D_reversed")]
        row = {"seed": seed}
        angles_ab: dict = {}
        angles_ad: dict = {}
        ratios: dict = {}
        for n in N_VALUES:
            ab = _angle_at_n(sa, sb, n)
            ad = _angle_at_n(sa, sd, n)
            angles_ab[str(n)] = ab
            angles_ad[str(n)] = ad
            ratios[str(n)] = float(ad / ab) if ab > 0 else float("nan")
        row["angle_A_vs_B_by_n"] = angles_ab
        row["angle_A_vs_D_by_n"] = angles_ad
        row["order_effect_ratio_by_n"] = ratios
        per_seed_pairs.append(row)
        print(f"  seed={seed}")
        print(f"    A vs B (different)   " + "  ".join(
            f"n={n}:{angles_ab[str(n)]:5.2f}" for n in N_VALUES))
        print(f"    A vs D (reversed)    " + "  ".join(
            f"n={n}:{angles_ad[str(n)]:5.2f}" for n in N_VALUES))
        print(f"    order effect ratio   " + "  ".join(
            f"n={n}:{ratios[str(n)]:.3f}" for n in N_VALUES))

    # Average across seeds
    across_seeds_ratio = {str(n): float(np.mean([r["order_effect_ratio_by_n"][str(n)]
                                                   for r in per_seed_pairs]))
                          for n in N_VALUES}
    across_seeds_diff_angle = {str(n): float(np.mean([r["angle_A_vs_B_by_n"][str(n)]
                                                     for r in per_seed_pairs]))
                               for n in N_VALUES}
    across_seeds_rev_angle = {str(n): float(np.mean([r["angle_A_vs_D_by_n"][str(n)]
                                                    for r in per_seed_pairs]))
                              for n in N_VALUES}
    print(f"\n  across-seed averages:")
    print(f"    mean A vs B by n:  " + "  ".join(
        f"n={n}:{across_seeds_diff_angle[str(n)]:5.2f}" for n in N_VALUES))
    print(f"    mean A vs D by n:  " + "  ".join(
        f"n={n}:{across_seeds_rev_angle[str(n)]:5.2f}" for n in N_VALUES))
    print(f"    mean order ratio:  " + "  ".join(
        f"n={n}:{across_seeds_ratio[str(n)]:.3f}" for n in N_VALUES))

    # ── (3) Accumulation rate sensitivity (B3-style) ───────────────────────
    print(f"\n[3/3] Accumulation-rate sensitivity to n (A vs B, per-cycle)...")
    accum_by_n: dict[str, list[dict]] = {str(n): [] for n in N_VALUES}
    for seed in SEEDS:
        snaps_a = accum_snapshots[(seed, "A_low")]
        snaps_b = accum_snapshots[(seed, "B_high")]
        for cycle_idx, (sa, sb) in enumerate(zip(snaps_a, snaps_b), start=1):
            for n in N_VALUES:
                ang = _angle_at_n(sa, sb, n)
                accum_by_n[str(n)].append({
                    "seed": seed, "cycle": cycle_idx, "angle": ang,
                })
    # Fit accumulation slope (angle vs cycle) per n, mean across seeds
    slope_by_n: dict[str, float] = {}
    for n in N_VALUES:
        rows = accum_by_n[str(n)]
        cycles = np.array([r["cycle"] for r in rows])
        angles = np.array([r["angle"] for r in rows])
        # Linear regression: angle = a*cycle + b
        slope = float(np.polyfit(cycles, angles, 1)[0])
        slope_by_n[str(n)] = slope
    print(f"  accumulation slope (deg per wake-sleep cycle):")
    for n in N_VALUES:
        print(f"    n={n}:  slope = {slope_by_n[str(n)]:+.3f} °/cycle")

    # ── Assemble ───────────────────────────────────────────────────────────
    principled_n = summary["n_var_99"]["max"]
    print(f"\n[recommendation]  principled n = max n_var_99 across seeds/streams = {principled_n}")
    print(f"                 (99% of variance captured; anything larger is noise)")

    out = {
        "config": {
            "hidden_dim": HIDDEN_DIM, "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM, "tau": TAU, "target_sr": SR_INIT,
            "alpha": ALPHA, "eta": ETA, "decay": DECAY,
            "idle_steps": IDLE_STEPS, "device": str(device),
            "seeds": SEEDS, "streams": {k: v for k, v in STREAMS.items()},
            "n_values_swept": N_VALUES,
        },
        "effective_dim": {
            "per_seed_stream": effective_dim_rows,
            "summary": summary,
        },
        "n_sensitivity_pairs": {
            "per_seed": per_seed_pairs,
            "across_seeds": {
                "mean_angle_A_vs_B_by_n": across_seeds_diff_angle,
                "mean_angle_A_vs_D_by_n": across_seeds_rev_angle,
                "mean_order_effect_ratio_by_n": across_seeds_ratio,
            },
        },
        "n_sensitivity_accumulation": {
            "slope_deg_per_cycle_by_n": slope_by_n,
            "per_cycle_measurements": accum_by_n,
        },
        "recommended_n": {
            "value": principled_n,
            "justification": ("smallest n such that ≥99% of the idle "
                              "trajectory's variance is captured across "
                              "every measured seed × stream configuration; "
                              "components beyond this are noise."),
        },
    }

    try:
        from src.experiments.subspace_dim import figures as fig_mod
        fig_mod.render(out)
    except Exception as e:
        print(f"\n(skipping figures: {e})")

    return out
