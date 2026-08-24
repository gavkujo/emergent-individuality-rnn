"""
Figures for subspace_dim.

Writes to <repo>/figures/subspace_dim/:
    fig_variance_spectra.png     — cumulative variance vs PC index across configs
    fig_angle_by_n.png           — pairwise angles vs n (A-vs-B, A-vs-D)
    fig_order_ratio_by_n.png     — order-effect ratio vs n
    fig_accumulation_by_n.png    — accumulation rate (slope) vs n
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
    out = os.path.join(repo_root, "figures", "subspace_dim")
    os.makedirs(out, exist_ok=True)
    return out


def _save(fig, name: str) -> str:
    path = os.path.join(_figures_dir(), name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {os.path.relpath(path)}")
    return path


def _fig_variance_spectra(results: dict) -> None:
    rows = results["effective_dim"]["per_seed_stream"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for row in rows:
        var_pct = np.array(row["variance_top10_pct"])
        cum = np.cumsum(var_pct)
        ax.plot(range(1, len(cum) + 1), cum, "o-", alpha=0.7,
                label=f"seed={row['seed']}, {row['stream']}")
    ax.axhline(99, color="gray", linestyle="--", linewidth=1.2,
               label="99% variance")
    ax.axhline(90, color="gray", linestyle=":", linewidth=1.2,
               label="90% variance")
    ax.axvline(2, color="#C44E52", linestyle="-", linewidth=1.5, alpha=0.6,
               label="n=2 (recommended)")
    ax.axvline(3, color="#8172B2", linestyle=":", linewidth=1.3,
               label="n=3 (legacy)")
    ax.set_xlabel("principal component index")
    ax.set_ylabel("cumulative variance (%)")
    ax.set_title("Idle-trajectory variance spectra — every config is essentially 2D")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(alpha=0.2)
    _save(fig, "fig_variance_spectra.png")


def _fig_angle_by_n(results: dict) -> None:
    across = results["n_sensitivity_pairs"]["across_seeds"]
    n_values = list(map(int, across["mean_angle_A_vs_B_by_n"].keys()))
    ab = [across["mean_angle_A_vs_B_by_n"][str(n)] for n in n_values]
    ad = [across["mean_angle_A_vs_D_by_n"][str(n)] for n in n_values]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(n_values, ab, "o-", color="#4C72B0", linewidth=2, markersize=6,
            label="A vs B (different content)")
    ax.plot(n_values, ad, "s--", color="#55A868", linewidth=2, markersize=6,
            label="A vs D (same tasks, reversed order)")
    ax.axvline(2, color="#C44E52", linestyle="-", linewidth=1.5, alpha=0.5,
               label="n=2 (recommended)")
    ax.axvline(3, color="#8172B2", linestyle=":", linewidth=1.3,
               label="n=3 (legacy)")
    ax.set_xlabel("n (number of principal directions)")
    ax.set_ylabel("subspace angle (°, mean across seeds)")
    ax.set_title("Reported subspace angle vs n — the metric is n-sensitive")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    _save(fig, "fig_angle_by_n.png")


def _fig_order_ratio_by_n(results: dict) -> None:
    across = results["n_sensitivity_pairs"]["across_seeds"]
    per_seed = results["n_sensitivity_pairs"]["per_seed"]
    n_values = list(map(int, across["mean_order_effect_ratio_by_n"].keys()))
    ratio_mean = [across["mean_order_effect_ratio_by_n"][str(n)] for n in n_values]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for row in per_seed:
        vals = [row["order_effect_ratio_by_n"][str(n)] for n in n_values]
        ax.plot(n_values, vals, "o-", alpha=0.35, color="#4C72B0",
                markersize=4, linewidth=1.0)
    ax.plot(n_values, ratio_mean, "o-", color="#4C72B0", markersize=8,
            linewidth=2.5, label="mean across seeds")
    ax.axvline(2, color="#C44E52", linestyle="-", linewidth=1.5, alpha=0.5,
               label="n=2 (recommended)")
    ax.axvline(3, color="#8172B2", linestyle=":", linewidth=1.3,
               label="n=3 (legacy)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0,
               label="no order effect (ratio = 1)")
    ax.set_xlabel("n")
    ax.set_ylabel("order-effect ratio (A-vs-D) / (A-vs-B)")
    ax.set_title("B6 order-effect ratio vs n — the '86%' number depends on n")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    _save(fig, "fig_order_ratio_by_n.png")


def _fig_accumulation_by_n(results: dict) -> None:
    slope = results["n_sensitivity_accumulation"]["slope_deg_per_cycle_by_n"]
    n_values = list(map(int, slope.keys()))
    slopes = [slope[str(n)] for n in n_values]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(n_values, slopes, color="#55A868", alpha=0.85, width=0.6)
    for n, s in zip(n_values, slopes):
        ax.text(n, s + max(slopes) * 0.02,
                f"{s:+.2f}", ha="center", va="bottom", fontsize=9)
    ax.axvline(2, color="#C44E52", linestyle="-", linewidth=1.5, alpha=0.5,
               label="n=2 (recommended)")
    ax.axvline(3, color="#8172B2", linestyle=":", linewidth=1.3,
               label="n=3 (legacy)")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("n")
    ax.set_ylabel("accumulation slope (° per wake-sleep cycle)")
    ax.set_title("B3 accumulation-rate vs n — direction (positive) is invariant")
    ax.legend(fontsize=9)
    _save(fig, "fig_accumulation_by_n.png")


def render(results: dict) -> None:
    print("\nRendering figures →", _figures_dir())
    _fig_variance_spectra(results)
    _fig_angle_by_n(results)
    _fig_order_ratio_by_n(results)
    _fig_accumulation_by_n(results)


def render_from_latest(name: str = "subspace_dim",
                       repo_root: Optional[str] = None) -> None:
    payload = load_latest(name, repo_root=repo_root)
    render(payload["results"])


if __name__ == "__main__":
    render_from_latest()
