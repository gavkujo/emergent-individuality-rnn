"""
Figures for sleep_order_multiseed.

    fig_b5_sleep_delta.png   — per-seed sleep effect (with − without) with
                                bootstrap CI, one panel per metric
    fig_b6_order_ratio.png   — per-seed order ratio (A-vs-D / A-vs-B) with
                                bootstrap CI, one panel per metric
    fig_b5_bars.png          — with vs without sleep per metric, bar chart
    fig_b6_bars.png          — A-vs-B vs A-vs-D per metric, bar chart
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
    out = os.path.join(repo_root, "figures", "sleep_order_multiseed")
    os.makedirs(out, exist_ok=True)
    return out


def _save(fig, name: str) -> str:
    path = os.path.join(_figures_dir(), name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {os.path.relpath(path)}")
    return path


def _fig_b5_sleep_delta(results: dict) -> None:
    seeds = results["config"]["seeds"]
    b5 = results["b5_sleep_effect"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, key, label in zip(axes,
                              ["angle_n2", "angle_n3", "l2"],
                              ["Δ subspace angle at n=2 (°)",
                               "Δ subspace angle at n=3 (°)",
                               "Δ L2 divergence"]):
        summ = b5["summary"][key]
        deltas = np.array(summ["delta_per_seed"])
        ds = summ["delta_stats"]

        # Per-seed dots
        jitter = np.random.default_rng(hash(key) & 0xFFFF).uniform(-0.15, 0.15, size=len(deltas))
        ax.scatter([0 + j for j in jitter], deltas, color="#4C72B0",
                   alpha=0.55, s=35, zorder=3)
        # Mean ± bootstrap CI
        ax.errorbar([0], [ds["mean_point_estimate"]],
                    yerr=[[ds["mean_point_estimate"] - ds["bootstrap_ci_lo"]],
                          [ds["bootstrap_ci_hi"] - ds["mean_point_estimate"]]],
                    fmt="s", color="#C44E52", markersize=11,
                    capsize=8, elinewidth=2, capthick=2, zorder=5)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.2)
        ax.set_xticks([])
        ax.set_ylabel(label)
        ax.set_title(f"Δ = {ds['mean_point_estimate']:+.3f}, "
                     f"CI [{ds['bootstrap_ci_lo']:+.3f}, "
                     f"{ds['bootstrap_ci_hi']:+.3f}]", fontsize=10)
        ax.grid(alpha=0.2)

    fig.suptitle(f"B5 sleep effect (with − without), {len(seeds)} seeds, "
                 f"A_low vs B_high after 5 cycles",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_b5_sleep_delta.png")


def _fig_b6_order_ratio(results: dict) -> None:
    seeds = results["config"]["seeds"]
    b6 = results["b6_order_effect"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, key, label in zip(axes,
                              ["angle_n2", "angle_n3", "l2"],
                              ["ratio, subspace angle at n=2",
                               "ratio, subspace angle at n=3",
                               "ratio, L2 divergence"]):
        summ = b6["summary"][key]
        ratios = np.array(summ["ratio_per_seed"])
        rs = summ["ratio_stats"]

        jitter = np.random.default_rng(hash(key) & 0xFFFF).uniform(-0.15, 0.15, size=len(ratios))
        ax.scatter([0 + j for j in jitter], ratios, color="#4C72B0",
                   alpha=0.55, s=35, zorder=3)
        ax.errorbar([0], [rs["mean_point_estimate"]],
                    yerr=[[rs["mean_point_estimate"] - rs["bootstrap_ci_lo"]],
                          [rs["bootstrap_ci_hi"] - rs["mean_point_estimate"]]],
                    fmt="s", color="#C44E52", markersize=11,
                    capsize=8, elinewidth=2, capthick=2, zorder=5)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2,
                   label="no order effect (ratio = 1)")
        ax.set_xticks([])
        ax.set_ylabel(label)
        ax.set_title(f"ratio = {rs['mean_point_estimate']:.3f}, "
                     f"CI [{rs['bootstrap_ci_lo']:.3f}, "
                     f"{rs['bootstrap_ci_hi']:.3f}]", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.2)

    fig.suptitle(f"B6 order effect (A-vs-D / A-vs-B), {len(seeds)} seeds",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_b6_order_ratio.png")


def _fig_b5_bars(results: dict) -> None:
    seeds = results["config"]["seeds"]
    b5 = results["b5_sleep_effect"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, key, label in zip(axes,
                              ["angle_n2", "angle_n3", "l2"],
                              ["subspace angle n=2 (°)",
                               "subspace angle n=3 (°)",
                               "L2 divergence"]):
        summ = b5["summary"][key]
        w = summ["with_sleep_mean"]
        no = summ["without_sleep_mean"]
        # Compute SEMs from per-seed
        with_arr = np.array([r["with_sleep"][key] for r in b5["per_seed"]])
        no_arr = np.array([r["without_sleep"][key] for r in b5["per_seed"]])
        w_sem = float(with_arr.std(ddof=1) / np.sqrt(len(with_arr)))
        no_sem = float(no_arr.std(ddof=1) / np.sqrt(len(no_arr)))

        bars = ax.bar(["without sleep", "with sleep"],
                      [no, w], yerr=[no_sem, w_sem],
                      color=["#aac4e0", "#4C72B0"], alpha=0.9,
                      capsize=6, error_kw=dict(linewidth=1.3))
        for bar, v in zip(bars, [no, w]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(no, w) * 0.02, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.suptitle(f"B5 sleep effect at cycle 5, {len(seeds)} seeds",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_b5_bars.png")


def _fig_b6_bars(results: dict) -> None:
    seeds = results["config"]["seeds"]
    b6 = results["b6_order_effect"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, key, label in zip(axes,
                              ["angle_n2", "angle_n3", "l2"],
                              ["subspace angle n=2 (°)",
                               "subspace angle n=3 (°)",
                               "L2 divergence"]):
        ab = np.array([r["A_vs_B"][key] for r in b6["per_seed"]])
        ad = np.array([r["A_vs_D"][key] for r in b6["per_seed"]])
        ab_sem = float(ab.std(ddof=1) / np.sqrt(len(ab)))
        ad_sem = float(ad.std(ddof=1) / np.sqrt(len(ad)))

        bars = ax.bar(["A vs B\n(different)", "A vs D\n(reversed)"],
                      [ab.mean(), ad.mean()],
                      yerr=[ab_sem, ad_sem],
                      color=["#C44E52", "#8172B2"], alpha=0.9,
                      capsize=6, error_kw=dict(linewidth=1.3))
        for bar, v in zip(bars, [ab.mean(), ad.mean()]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(ab.mean(), ad.mean()) * 0.02, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.suptitle(f"B6 order effect at cycle 5, {len(seeds)} seeds",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_b6_bars.png")


def render(results: dict) -> None:
    print("\nRendering figures →", _figures_dir())
    _fig_b5_sleep_delta(results)
    _fig_b5_bars(results)
    _fig_b6_order_ratio(results)
    _fig_b6_bars(results)


def render_from_latest(name: str = "sleep_order_multiseed",
                       repo_root: Optional[str] = None) -> None:
    payload = load_latest(name, repo_root=repo_root)
    render(payload["results"])


if __name__ == "__main__":
    render_from_latest()
