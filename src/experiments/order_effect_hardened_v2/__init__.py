"""
Order-effect v2: isolate ORDER from RECENCY via random-permutation streams.

Motivation. `order_effect_hardened_v1` reported ratio 1.70 with
CI [1.60, 1.84], opposite direction of the paper's B6 (0.86). Finding
`05_order_effect_hardened_v1.md` traced this to a sampling confound:
`sample_specs` deterministically orders freqs low → high within each
stream, so `S_rev` always ends at the bottom bin while `S_fwd` and `T`
always end at the top bin. Under any recency-biased identity, that
alone yields ratio > 1.

Fix (this experiment):

    Per seed:
      1. Sample K=5 freqs from log-uniform [0.5, 20] (base "content").
      2. `S_perm_A` = these K freqs in a random permutation.
      3. `S_perm_B` = the SAME K freqs in a DIFFERENT random permutation.
      4. `T_perm`   = K FRESH freqs (from the same distribution) in a
         random permutation.

    Last-freq of `S_perm_A`, `S_perm_B`, and `T_perm` are all draws
    from the same underlying frequency distribution — no systematic
    recency bias.

Measurement:

    ratio = angle(S_perm_A, S_perm_B) / angle(S_perm_A, T_perm)

Verdict:

    ratio < 1 with CI excluding 1 → order matters LESS than
        content variation (paper's original framing).
    ratio > 1 with CI excluding 1 → order matters MORE than
        content variation (surprising, strong claim).
    ratio ≈ 1 with CI including 1 → order and content variation
        contribute equally (paper's order-as-first-class claim
        collapses; identity is a residue of "which tasks did the
        network see," not "in what order").

We also record the actual last-freq gap for each pair as a
per-seed diagnostic, so a reviewer can see the recency confound
is really neutralised.
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


NAME = "order_effect_hardened_v2"
DESCRIPTION = ("Order-effect v2: random-permutation streams isolate ORDER "
               "from RECENCY. n=30 seeds. Per seed: two permutations of "
               "the same K=5 freqs vs a permutation of a fresh K=5 freqs. "
               "Bootstrap CI on the ratio.")
DETAILS = ("v2: LeakyRNN (hidden_dim=256, τ=5, target_sr=0.95, α=0.8). "
           "Per-seed: 5 base freqs sampled log-uniform [0.5, 20]; two "
           "random permutations produce S_perm_A and S_perm_B (same "
           "content, different order); a fresh 5-freq sample produces "
           "T_perm (different content, same distribution). Wake_phase "
           "200 steps, sleep_phase 600 steps, η=0.01, decay=0.001, ACh "
           "gate 0.3. Idle 400 steps. Bootstrap 2000 resamples for 95% "
           "CI. Also records last-freq gap per pair as a recency-check "
           "diagnostic.")
VERSION = "v2"


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
    # Seed offsets for reproducible per-seed stream generation
    "base_freqs_seed_offset":   300000,
    "unrelated_freqs_seed_offset": 400000,
    "perm_A_seed_offset":       500000,
    "perm_B_seed_offset":       600000,
    "perm_T_seed_offset":       700000,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def _permute(specs, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(specs))
    return [specs[i] for i in idx]


# ── Per-seed body ────────────────────────────────────────────────────────────

def per_seed(seed: int, device, config: dict) -> dict:
    L = config["stream_length"]
    # Draw the base content — same K freqs for A and B, different for T
    base_content   = sample_specs(config["task_family"], L,
                                   seed=config["base_freqs_seed_offset"] + seed)
    unrelated_content = sample_specs(config["task_family"], L,
                                     seed=config["unrelated_freqs_seed_offset"] + seed)

    stream_A = _permute(base_content, config["perm_A_seed_offset"] + seed)
    stream_B = _permute(base_content, config["perm_B_seed_offset"] + seed)
    stream_T = _permute(unrelated_content, config["perm_T_seed_offset"] + seed)

    # Rare: if A and B are the identical permutation (probability 1/L!) reroll B
    if [s.label for s in stream_A] == [s.label for s in stream_B]:
        stream_B = _permute(base_content, config["perm_B_seed_offset"] + seed + 10000)

    def _train_one(stream):
        m = _fresh_leakyrnn(seed, config, device)
        r = _train(m, stream, config, device)
        idle = r["final_idle"]
        diag = r["sleep_diagnostics"]
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return idle, diag

    idle_A, diag_A = _train_one(stream_A)
    idle_B, diag_B = _train_one(stream_B)
    idle_T, diag_T = _train_one(stream_T)

    # Last-freq diagnostics — check the recency confound is neutralised
    last_A = float(stream_A[-1].params["freq"])
    last_B = float(stream_B[-1].params["freq"])
    last_T = float(stream_T[-1].params["freq"])
    gap_AB_log = float(abs(np.log(last_A) - np.log(last_B)))
    gap_AT_log = float(abs(np.log(last_A) - np.log(last_T)))

    ang_n2_AB = subspace_angle(idle_A, idle_B, n=2)
    ang_n2_AT = subspace_angle(idle_A, idle_T, n=2)
    ang_n3_AB = subspace_angle(idle_A, idle_B, n=3)
    ang_n3_AT = subspace_angle(idle_A, idle_T, n=3)
    l2_AB = l2_divergence(idle_A, idle_B)
    l2_AT = l2_divergence(idle_A, idle_T)

    def _ratio(a, b): return float(a / b) if b > 0 else float("nan")

    return {
        "stream_A": [s.to_dict() for s in stream_A],
        "stream_B": [s.to_dict() for s in stream_B],
        "stream_T": [s.to_dict() for s in stream_T],
        "last_freqs": {"A": last_A, "B": last_B, "T": last_T,
                       "gap_AB_log": gap_AB_log,
                       "gap_AT_log": gap_AT_log},
        "distances": {
            "angle_n2_A_vs_B": float(ang_n2_AB),
            "angle_n2_A_vs_T": float(ang_n2_AT),
            "angle_n3_A_vs_B": float(ang_n3_AB),
            "angle_n3_A_vs_T": float(ang_n3_AT),
            "l2_A_vs_B":       float(l2_AB),
            "l2_A_vs_T":       float(l2_AT),
        },
        "ratios": {
            "angle_n2": _ratio(ang_n2_AB, ang_n2_AT),
            "angle_n3": _ratio(ang_n3_AB, ang_n3_AT),
            "l2":       _ratio(l2_AB,     l2_AT),
        },
        "sleep_diagnostics": {
            "A_final_sr": float(diag_A[-1]["post_sr"]),
            "B_final_sr": float(diag_B[-1]["post_sr"]),
            "T_final_sr": float(diag_T[-1]["post_sr"]),
        },
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def _verdict(ci: dict, null_value: float, label: str) -> str:
    if ci["ci_hi"] < null_value:
        return f"{label}: BELOW {null_value:.3f} (order matters LESS than content)"
    if ci["ci_lo"] > null_value:
        return f"{label}: ABOVE {null_value:.3f} (order matters MORE than content)"
    return f"{label}: INCONCLUSIVE (CI crosses {null_value:.3f})"


def aggregate(per_seed_results: dict[int, dict], config: dict) -> dict:
    seeds = sorted(per_seed_results.keys())
    n_boot = config["bootstrap_resamples"]
    ci_lvl = config["bootstrap_ci"]

    r = per_seed_results

    # Recency-check diagnostics
    gap_AB_log = np.array([r[s]["last_freqs"]["gap_AB_log"] for s in seeds])
    gap_AT_log = np.array([r[s]["last_freqs"]["gap_AT_log"] for s in seeds])
    last_freq_diagnostic = {
        "mean_gap_AB_log": float(gap_AB_log.mean()),
        "mean_gap_AT_log": float(gap_AT_log.mean()),
        "delta_gap_AB_minus_AT_log": float((gap_AB_log - gap_AT_log).mean()),
        "note": ("If the mean of gap_AB_log ≈ mean of gap_AT_log, the "
                 "recency confound of v1 is neutralised. A large delta "
                 "here would mean the sampling is still biasing which "
                 "stream ends near which frequency."),
    }

    # Distances
    def _bs(key, name, seed_offset):
        arr = np.array([r[s]["distances"][key] for s in seeds])
        return bootstrap_ci_mean(arr, n_boot, ci_lvl, seed=seed_offset)

    distances = {
        "angle_n2_A_vs_B": _bs("angle_n2_A_vs_B", "AB_n2", 100),
        "angle_n2_A_vs_T": _bs("angle_n2_A_vs_T", "AT_n2", 101),
        "angle_n3_A_vs_B": _bs("angle_n3_A_vs_B", "AB_n3", 102),
        "angle_n3_A_vs_T": _bs("angle_n3_A_vs_T", "AT_n3", 103),
        "l2_A_vs_B":       _bs("l2_A_vs_B", "AB_l2", 104),
        "l2_A_vs_T":       _bs("l2_A_vs_T", "AT_l2", 105),
    }

    # Ratios
    ratio_n2 = np.array([r[s]["ratios"]["angle_n2"] for s in seeds])
    ratio_n3 = np.array([r[s]["ratios"]["angle_n3"] for s in seeds])
    ratio_l2 = np.array([r[s]["ratios"]["l2"] for s in seeds])

    ci_n2 = bootstrap_ci_mean(ratio_n2, n_boot, ci_lvl, seed=200)
    ci_n3 = bootstrap_ci_mean(ratio_n3, n_boot, ci_lvl, seed=201)
    ci_l2 = bootstrap_ci_mean(ratio_l2, n_boot, ci_lvl, seed=202)

    return {
        "n_seeds": len(seeds),
        "stream_length": config["stream_length"],
        "recency_diagnostic": last_freq_diagnostic,
        "distances": distances,
        "ratios": {
            "angle_n2": {"bootstrap_ci": ci_n2, "per_seed": ratio_n2.tolist(),
                        "n_below_one": int((ratio_n2 < 1.0).sum())},
            "angle_n3": {"bootstrap_ci": ci_n3, "per_seed": ratio_n3.tolist(),
                        "n_below_one": int((ratio_n3 < 1.0).sum())},
            "l2":       {"bootstrap_ci": ci_l2, "per_seed": ratio_l2.tolist(),
                        "n_below_one": int((ratio_l2 < 1.0).sum())},
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
        name=NAME, version=VERSION, seeds=seeds,
        per_seed_fn=per_seed, aggregate_fn=aggregate,
        config=CONFIG, device=device,
        description=DESCRIPTION, details=DETAILS, verbose=True,
    )

    a = final["aggregated"]
    print("\n" + "=" * 60)
    print(f"ORDER_EFFECT_HARDENED v2 summary (random-permutation streams)")
    print("=" * 60)
    print(f"n_seeds = {a['n_seeds']}, stream_length = {a['stream_length']}")
    rec = a["recency_diagnostic"]
    print(f"\n  Recency check (in log-Hz):")
    print(f"    mean |log(last_A) - log(last_B)| = {rec['mean_gap_AB_log']:.3f}")
    print(f"    mean |log(last_A) - log(last_T)| = {rec['mean_gap_AT_log']:.3f}")
    print(f"    difference (AB - AT)              = {rec['delta_gap_AB_minus_AT_log']:+.3f}")
    print(f"    (small delta → recency confound is neutralised)")
    print(f"\n  Absolute distances:")
    for k, v in a["distances"].items():
        print(f"    {k:22s} = {v['mean']:6.2f}  95% CI [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]")
    print(f"\n  Ratios (order-vs-order over order-vs-content):")
    for m in ["angle_n2", "angle_n3", "l2"]:
        r = a["ratios"][m]
        ci = r["bootstrap_ci"]
        print(f"    {m:8s} ratio = {ci['mean']:.3f}  "
              f"95% CI [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]  "
              f"below 1: {r['n_below_one']}/{a['n_seeds']}")
    print(f"\n  Verdicts:")
    for v in a["verdicts"]:
        print(f"    - {v}")
    return None
