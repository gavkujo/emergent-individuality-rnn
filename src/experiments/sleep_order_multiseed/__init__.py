"""
Multi-seed test of the B5 (sleep-effect) and B6 (order-effect) claims.

The workshop paper reports:
    B5 sleep effect: with-sleep vs without-sleep, cycle 5, A_low vs B_high.
                     Paper claims sleep amplifies L2 divergence (+10%) and
                     reduces subspace angle (dissociation).
    B6 order effect: A_low vs D_reversed (same tasks, different order) vs
                     A_low vs B_high (different content). Paper claims
                     ~86% of the different-content divergence.

Both were originally reported at n=1 seed. Given that B3 accumulation
turned out to be seed-variance-dominated (see finding
03_accumulation_rejected_multiseed.md), we replicate B5 and B6 at
n=10 seeds with bootstrap CIs.

Efficient design: A_with_sleep and B_with_sleep networks are used by
BOTH B5 and B6, so we train each once per seed and reuse.

Per seed we train 5 networks:
    A_ws:  wake+sleep on STREAM_A
    B_ws:  wake+sleep on STREAM_B
    D_ws:  wake+sleep on STREAM_D  (for B6)
    A_ns:  wake-only on STREAM_A   (for B5 without-sleep)
    B_ns:  wake-only on STREAM_B   (for B5 without-sleep)

Metrics use n=2 (post-hardening default) for subspace_angle. Also
n=3 for a legacy-comparison column.
"""

from __future__ import annotations

import numpy as np
import torch

from src.model import LeakyRNN
from src.train import make_sine_task, wake_phase, sleep_phase, run_idle


NAME = "sleep_order_multiseed"
DESCRIPTION = ("Multi-seed replication of B5 (sleep effect) and B6 (order "
               "effect) claims. 10 seeds, bootstrap CIs on the effect sizes.")
DETAILS = ("v1: LeakyRNN (seeds 42..51, tau=5, target_sr=0.95, α=0.8, "
           "s_in=s_out=0.1). Wake_phase 200 steps, sleep_phase 600 steps, "
           "η=0.01, decay=0.001, ACh gate 0.3. Idle 300 steps. Streams "
           "A_low=[1,1.5,2,1,2], B_high=[8,12,6,8,6], D_reversed=[2,1.5,"
           "1,2,1] (STREAM_A reversed). Subspace angle at n=2 and n=3; "
           "L2 divergence. Bootstrap 2000 resamples for 95% CI.")
VERSION = "v1"


INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 16, 128, 16
TAU, SR_INIT, ALPHA = 5.0, 0.95, 0.8
ETA, DECAY = 0.01, 0.001
WAKE_STEPS, SLEEP_STEPS, IDLE_STEPS = 200, 600, 300

STREAM_A = [1.0, 1.5, 2.0, 1.0, 2.0]
STREAM_B = [8.0, 12.0, 6.0, 8.0, 6.0]
STREAM_D = [2.0, 1.5, 1.0, 2.0, 1.0]

SEEDS = list(range(42, 52))
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
    return float(np.linalg.norm(sa - sb, axis=-1).mean())


# ── Training ────────────────────────────────────────────────────────────────

def _train_stream(seed: int, stream: list[float], device, with_sleep: bool) -> np.ndarray:
    """Train a fresh LeakyRNN on `stream`. If `with_sleep`, run
    sleep_phase after each wake_phase. Return the final idle trajectory.

    Explicitly releases the model and clears MPS cache before returning,
    to avoid slowdown from accumulated device memory across sequential
    trainings.
    """
    torch.manual_seed(seed)
    m = LeakyRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
                 tau=TAU, alpha=ALPHA, target_sr=SR_INIT).to(device)
    for i, freq in enumerate(stream):
        wake_phase(m, *make_sine_task(freq, seed=i, device=device),
                   steps=WAKE_STEPS, lr=3e-3, ach_gate=1.0)
        if with_sleep:
            sleep_phase(m, sleep_steps=SLEEP_STEPS, eta=ETA, decay=DECAY,
                        ach_gate=0.3, device=device)
    traj = run_idle(m, steps=IDLE_STEPS, device=device)
    del m
    if str(device) == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return traj


# ── Bootstrap over paired observations ──────────────────────────────────────

def _bootstrap_stats(per_seed: np.ndarray, n_resamples: int,
                     ci_level: float,
                     rng: np.random.Generator) -> dict:
    """Bootstrap median and CI of the mean over a 1D array of per-seed values."""
    n = len(per_seed)
    means = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[r] = per_seed[idx].mean()
    lo = 0.5 - ci_level / 2.0
    hi = 0.5 + ci_level / 2.0
    return {
        "mean_point_estimate": float(per_seed.mean()),
        "std_across_seeds":    float(per_seed.std(ddof=1)),
        "sem":                  float(per_seed.std(ddof=1) / np.sqrt(n)),
        "bootstrap_median":    float(np.median(means)),
        "bootstrap_ci_lo":     float(np.quantile(means, lo)),
        "bootstrap_ci_hi":     float(np.quantile(means, hi)),
        "bootstrap_ci_level":  ci_level,
        "bootstrap_resamples": n_resamples,
    }


# ── Main experiment ─────────────────────────────────────────────────────────

def run(device) -> dict:
    torch.set_default_dtype(torch.float32)
    print(f"\nsleep_order_multiseed: seeds={SEEDS}, device={device}")

    # Storage
    trajectories: dict[tuple[int, str], np.ndarray] = {}

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\nTraining 5 networks per seed (3 with-sleep, 2 without)...")
    for seed in SEEDS:
        print(f"\n  [seed {seed}]", flush=True)
        print(f"    A_low   (with sleep) ...", flush=True)
        trajectories[(seed, "A_ws")] = _train_stream(seed, STREAM_A, device, with_sleep=True)
        print(f"    B_high  (with sleep) ...", flush=True)
        trajectories[(seed, "B_ws")] = _train_stream(seed, STREAM_B, device, with_sleep=True)
        print(f"    D_rev   (with sleep) ...", flush=True)
        trajectories[(seed, "D_ws")] = _train_stream(seed, STREAM_D, device, with_sleep=True)
        print(f"    A_low   (no sleep) ...", flush=True)
        trajectories[(seed, "A_ns")] = _train_stream(seed, STREAM_A, device, with_sleep=False)
        print(f"    B_high  (no sleep) ...", flush=True)
        trajectories[(seed, "B_ns")] = _train_stream(seed, STREAM_B, device, with_sleep=False)

    # ── B5: sleep effect ───────────────────────────────────────────────────
    print("\n\nComputing B5 sleep effect (with vs without, A vs B)...")
    b5_rows = []
    for seed in SEEDS:
        A_ws, B_ws = trajectories[(seed, "A_ws")], trajectories[(seed, "B_ws")]
        A_ns, B_ns = trajectories[(seed, "A_ns")], trajectories[(seed, "B_ns")]
        row = {
            "seed": seed,
            "with_sleep": {
                "angle_n2": _subspace_angle_at_n(A_ws, B_ws, 2),
                "angle_n3": _subspace_angle_at_n(A_ws, B_ws, 3),
                "l2":       _l2_divergence(A_ws, B_ws),
            },
            "without_sleep": {
                "angle_n2": _subspace_angle_at_n(A_ns, B_ns, 2),
                "angle_n3": _subspace_angle_at_n(A_ns, B_ns, 3),
                "l2":       _l2_divergence(A_ns, B_ns),
            },
        }
        row["sleep_delta"] = {
            k: row["with_sleep"][k] - row["without_sleep"][k]
            for k in ["angle_n2", "angle_n3", "l2"]
        }
        b5_rows.append(row)
        print(f"  seed={seed}: "
              f"Δ angle_n2 = {row['sleep_delta']['angle_n2']:+6.2f}, "
              f"Δ angle_n3 = {row['sleep_delta']['angle_n3']:+6.2f}, "
              f"Δ l2 = {row['sleep_delta']['l2']:+.3f}")

    # ── B6: order effect ───────────────────────────────────────────────────
    print("\nComputing B6 order effect (A vs D relative to A vs B, with sleep)...")
    b6_rows = []
    for seed in SEEDS:
        A, B, D = trajectories[(seed, "A_ws")], trajectories[(seed, "B_ws")], trajectories[(seed, "D_ws")]
        row = {
            "seed": seed,
            "A_vs_B": {
                "angle_n2": _subspace_angle_at_n(A, B, 2),
                "angle_n3": _subspace_angle_at_n(A, B, 3),
                "l2":       _l2_divergence(A, B),
            },
            "A_vs_D": {
                "angle_n2": _subspace_angle_at_n(A, D, 2),
                "angle_n3": _subspace_angle_at_n(A, D, 3),
                "l2":       _l2_divergence(A, D),
            },
        }
        row["order_ratio"] = {
            k: (row["A_vs_D"][k] / row["A_vs_B"][k]) if row["A_vs_B"][k] > 0
               else float("nan")
            for k in ["angle_n2", "angle_n3", "l2"]
        }
        b6_rows.append(row)
        print(f"  seed={seed}: "
              f"ratio n2 = {row['order_ratio']['angle_n2']:.3f}, "
              f"n3 = {row['order_ratio']['angle_n3']:.3f}, "
              f"l2 = {row['order_ratio']['l2']:.3f}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    rng = np.random.default_rng(0)

    print(f"\n\nAggregate over {len(SEEDS)} seeds:")

    b5_summary = {}
    for key in ["angle_n2", "angle_n3", "l2"]:
        with_vals = np.array([r["with_sleep"][key] for r in b5_rows])
        without_vals = np.array([r["without_sleep"][key] for r in b5_rows])
        delta_vals = with_vals - without_vals
        summary = {
            "with_sleep_mean": float(with_vals.mean()),
            "without_sleep_mean": float(without_vals.mean()),
            "delta_stats": _bootstrap_stats(delta_vals, BOOTSTRAP_RESAMPLES,
                                            BOOTSTRAP_CI, rng),
            "delta_per_seed": delta_vals.tolist(),
            "seeds_with_sleep_larger": int(np.sum(delta_vals > 0)),
            "n_total": len(SEEDS),
        }
        b5_summary[key] = summary
        d = summary["delta_stats"]
        verdict = ("SLEEP AMPLIFIES divergence (CI > 0)" if d["bootstrap_ci_lo"] > 0 else
                   "SLEEP REDUCES divergence (CI < 0)" if d["bootstrap_ci_hi"] < 0 else
                   "INCONCLUSIVE (CI includes 0)")
        print(f"  B5 {key}:  Δ = {d['mean_point_estimate']:+.3f}  "
              f"CI [{d['bootstrap_ci_lo']:+.3f}, {d['bootstrap_ci_hi']:+.3f}]  "
              f"sign {summary['seeds_with_sleep_larger']}/{len(SEEDS)}")
        print(f"       {verdict}")

    b6_summary = {}
    for key in ["angle_n2", "angle_n3", "l2"]:
        ab_vals = np.array([r["A_vs_B"][key] for r in b6_rows])
        ad_vals = np.array([r["A_vs_D"][key] for r in b6_rows])
        ratio_vals = ad_vals / ab_vals
        summary = {
            "A_vs_B_mean": float(ab_vals.mean()),
            "A_vs_D_mean": float(ad_vals.mean()),
            "ratio_stats": _bootstrap_stats(ratio_vals, BOOTSTRAP_RESAMPLES,
                                            BOOTSTRAP_CI, rng),
            "ratio_per_seed": ratio_vals.tolist(),
            "seeds_with_ratio_below_1":
                int(np.sum(ratio_vals < 1.0)),
            "n_total": len(SEEDS),
        }
        b6_summary[key] = summary
        r_ = summary["ratio_stats"]
        verdict = ("ORDER EFFECT PRESENT (CI < 1)" if r_["bootstrap_ci_hi"] < 1.0 else
                   "NO ORDER EFFECT (CI > 1)" if r_["bootstrap_ci_lo"] > 1.0 else
                   "INCONCLUSIVE (CI includes 1)")
        print(f"  B6 {key}:  ratio = {r_['mean_point_estimate']:.3f}  "
              f"CI [{r_['bootstrap_ci_lo']:.3f}, {r_['bootstrap_ci_hi']:.3f}]  "
              f"below 1: {summary['seeds_with_ratio_below_1']}/{len(SEEDS)}")
        print(f"       {verdict}")

    out = {
        "config": {
            "hidden_dim": HIDDEN_DIM, "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM, "tau": TAU, "target_sr": SR_INIT,
            "alpha": ALPHA, "eta": ETA, "decay": DECAY,
            "wake_steps": WAKE_STEPS, "sleep_steps": SLEEP_STEPS,
            "idle_steps": IDLE_STEPS,
            "seeds": SEEDS, "stream_A_low": STREAM_A,
            "stream_B_high": STREAM_B, "stream_D_reversed": STREAM_D,
            "device": str(device),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_ci_level": BOOTSTRAP_CI,
        },
        "b5_sleep_effect": {
            "per_seed": b5_rows,
            "summary": b5_summary,
        },
        "b6_order_effect": {
            "per_seed": b6_rows,
            "summary": b6_summary,
        },
    }

    try:
        from src.experiments.sleep_order_multiseed import figures as fig_mod
        fig_mod.render(out)
    except Exception as e:
        print(f"\n(skipping figures: {e})")

    return out
