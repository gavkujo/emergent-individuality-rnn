"""
Figures for accumulation_multiseed. Written to <repo>/figures/
accumulation_multiseed/.

  fig_curves_by_metric.png — per-cycle mean ± SEM across seeds, one panel
                             per metric (subspace angle n=2, n=3, L2)
  fig_slopes_bootstrap.png — per-metric slope estimate with bootstrap
                             95% CI, plus the individual per-seed slopes
                             overlaid as dots
  fig_seed_curves.png      — every seed's cycle-1..5 curve at n=2, to
                             visualise seed variance directly
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.results_io import load_latest


plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})


def _figures_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    out = os.path.join(repo_root, "figures", "accumulation_multiseed")
    os.makedirs(out, exist_ok=True)
    return out


def _save(fig, name: str) -> str:
    path = os.path.join(_figures_dir(), name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {os.path.relpath(path)}")
    return path


def _fig_curves_by_metric(results: dict) -> None:
    cycles = np.array(results["cycles"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (key, label, colour) in zip(axes, [
        ("angle_n2", "subspace angle at n=2 (°)", "#4C72B0"),
        ("angle_n3", "subspace angle at n=3 (°)", "#8172B2"),
        ("l2",       "L2 divergence",              "#C44E52"),
    ]):
        summ = results["metric_summaries"][key]
        m = np.array(summ["per_cycle_mean"])
        s = np.array(summ["per_cycle_sem"])
        ax.errorbar(cycles, m, yerr=s, fmt="o-", color=colour,
                    linewidth=2, markersize=7, capsize=4,
                    label="mean ± SEM across seeds")

        # Also plot each seed lightly
        per_seed = np.array(results["per_seed_curves"][key])
        for seed_row in per_seed:
            ax.plot(cycles, seed_row, "-", color=colour, alpha=0.15,
                    linewidth=0.8)

        ci = summ["bootstrap_slope"]
        slope_str = (f"slope={summ['mean_curve_slope']:+.3f}, "
                     f"95% CI [{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]")
        ax.set_xlabel("wake-sleep cycle")
        ax.set_ylabel(label)
        ax.set_title(slope_str, fontsize=10)
        ax.set_xticks(cycles)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)

    fig.suptitle(f"B3 accumulation, {len(results['config']['seeds'])} seeds, "
                 f"A_low vs B_high", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_curves_by_metric.png")


def _fig_slopes_bootstrap(results: dict) -> None:
    keys = ["angle_n2", "angle_n3", "l2"]
    labels = ["angle (n=2)", "angle (n=3)", "L2"]
    fig, ax = plt.subplots(figsize=(7, 4.4))

    x = np.arange(len(keys))
    for i, key in enumerate(keys):
        summ = results["metric_summaries"][key]
        ci = summ["bootstrap_slope"]
        med = ci["median"]
        lo, hi = ci["ci_lo"], ci["ci_hi"]

        # CI bar
        ax.errorbar([i], [med], yerr=[[med - lo], [hi - med]], fmt="s",
                    color="#4C72B0", markersize=9, capsize=8,
                    elinewidth=2, capthick=2)
        # Per-seed slopes scattered
        per_seed_slopes = summ["per_seed_slope"]
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, size=len(per_seed_slopes))
        ax.scatter([i + j for j in jitter], per_seed_slopes,
                   color="#4C72B0", alpha=0.4, s=25, zorder=3)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.2,
               label="null (no accumulation)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("slope (metric per wake-sleep cycle)")
    ax.set_title("Slope estimates with bootstrap 95% CI and per-seed points")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    _save(fig, "fig_slopes_bootstrap.png")


def _fig_seed_curves(results: dict) -> None:
    cycles = np.array(results["cycles"])
    curves = np.array(results["per_seed_curves"]["angle_n2"])
    seeds = results["config"]["seeds"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, seed_row in enumerate(curves):
        colour = plt.cm.viridis(i / max(1, len(curves) - 1))
        ax.plot(cycles, seed_row, "o-", color=colour, alpha=0.8,
                markersize=5, linewidth=1.2, label=f"seed {seeds[i]}")

    mean = curves.mean(axis=0)
    ax.plot(cycles, mean, "-", color="black", linewidth=3, alpha=0.9,
            label="mean")
    ax.set_xlabel("wake-sleep cycle")
    ax.set_ylabel("subspace angle at n=2 (°)")
    ax.set_title("Per-seed accumulation curves (n=2)")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.set_xticks(cycles)
    ax.grid(alpha=0.2)
    _save(fig, "fig_seed_curves.png")


def render(results: dict) -> None:
    print("\nRendering figures →", _figures_dir())
    _fig_curves_by_metric(results)
    _fig_slopes_bootstrap(results)
    _fig_seed_curves(results)


def render_from_latest(name: str = "accumulation_multiseed",
                       repo_root: Optional[str] = None) -> None:
    payload = load_latest(name, repo_root=repo_root)
    render(payload["results"])


if __name__ == "__main__":
    render_from_latest()
