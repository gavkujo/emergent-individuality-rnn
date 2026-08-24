"""
Figures for the α_init experiment.

Reads a `results` dict (from `run`) or the latest JSON, writes PNGs to
<repo>/figures/alpha_init/.

Figures produced:
    fig_scale_search.png       — ρ(A(0)) vs s across the search grid
    fig_rho_sweep_at_sstar.png — ρ(A(α)) at s* with α* marked and (E.ii)
                                 shading of the region where (E.ii) fails
    fig_drift_bars.png         — ρ(W_rec) and ρ(A(α_eval)) init vs final,
                                 for both legacy and principled configs
    fig_linearisation.png      — ‖h‖ mean/max, tanh'(z) mean/min for both
                                 configs (post-training idle)
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
    """<repo>/figures/alpha_init/. From this file at repo/src/experiments/
    alpha_init/figures.py, the repo root is 3 dirname()s up."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    out = os.path.join(repo_root, "figures", "alpha_init")
    os.makedirs(out, exist_ok=True)
    return out


def _save(fig, name: str) -> str:
    path = os.path.join(_figures_dir(), name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {os.path.relpath(path)}")
    return path


# ── Figure builders ───────────────────────────────────────────────────────────

def _fig_scale_search(results: dict) -> None:
    ss = results["scale_search"]
    s_grid = np.array(ss["s_grid"])
    grid_res = ss["grid_results"]
    rho_A0 = np.array([r["rho_A_at_0"] for r in grid_res])
    monos = np.array([r["monotone_non_increasing"] for r in grid_res], dtype=bool)
    s_star = ss["s_star"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(s_grid, rho_A0, "-", color="#4C72B0", linewidth=1.5,
            label="ρ(A(0)) across scales")
    ax.scatter(s_grid[monos], rho_A0[monos], color="#55A868", s=25,
               label="monotone at that scale", zorder=5)
    ax.scatter(s_grid[~monos], rho_A0[~monos], color="#C44E52", s=25,
               label="non-monotone at that scale", zorder=5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2,
               label="(H2) threshold: ρ(A(0)) = 1")
    if s_star is not None:
        ax.axvline(s_star, color="#8172B2", linestyle="-", linewidth=1.5,
                   label=f"s* = {s_star:.4f}")
    ax.axvline(results["config"]["legacy_scale"], color="black",
               linestyle=":", linewidth=1.3,
               label=f"legacy scale s={results['config']['legacy_scale']}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("joint scale s applied to W_in and W_out")
    ax.set_ylabel("ρ(A(0))")
    ax.set_title("Scale search: smallest s satisfying the α theorem")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2, which="both")
    _save(fig, "fig_scale_search.png")


def _fig_rho_sweep_at_sstar(results: dict) -> None:
    ap = results["principled_alpha"]
    sweep = ap["monotonicity_sweep"]
    alphas = np.array(sweep["alphas"])
    rhos = np.array(sweep["rho_A"])
    alpha_star = ap["alpha_star"]
    rho_star = ap["rho_A_at_alpha_star"]
    s_star = ap["s_star"]

    # (E.ii) inner product across α
    per_alpha = ap["E_diagnostics"]["per_alpha"]
    eii = np.array([d["E_ii_inner_product_real"] for d in per_alpha])
    eii_alphas = np.array([d["alpha"] for d in per_alpha])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(alphas, rhos, "o-", color="#4C72B0", linewidth=2, markersize=5)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1.2,
                label="edge of chaos (ρ = 1)")
    ax1.axvline(alpha_star, color="#C44E52", linestyle="-", linewidth=1.5,
                label=f"α* = {alpha_star:.4f}")
    ax1.scatter([alpha_star], [rho_star], color="#C44E52", s=60, zorder=5)
    ax1.set_ylabel("ρ(A(α))")
    ax1.set_title(f"ρ(A(α)) at s* = {s_star:.4f} — theorem cleanly applies")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)

    ax2.plot(eii_alphas, eii, "o-", color="#DD8452", linewidth=2, markersize=5)
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("α")
    ax2.set_ylabel(r"$\mathrm{Re}(\overline{\lambda_A} u^\top W_{in}W_{out}\,v)$")
    ax2.set_title("(E.ii) inner product — > 0 ⇒ ρ(A(α)) strictly decreasing", fontsize=10)
    ax2.grid(alpha=0.2)
    _save(fig, "fig_rho_sweep_at_sstar.png")


def _fig_drift_bars(results: dict) -> None:
    cfgs = ["legacy\n(s=0.1, α=0.8)",
            f"principled\n(s={results['principled_alpha']['s_star']:.3f}, "
            f"α={results['principled_alpha']['alpha_star']:.3f})"]
    drift_L = results["drift"]["legacy_config"]
    drift_P = results["drift"]["principled_config"]

    rho_W_init = [drift_L["rho_W_rec"]["init"], drift_P["rho_W_rec"]["init"]]
    rho_W_final = [drift_L["rho_W_rec"]["final"], drift_P["rho_W_rec"]["final"]]
    rho_A_init = [drift_L["rho_A_at_alpha_eval"]["init"],
                  drift_P["rho_A_at_alpha_eval"]["init"]]
    rho_A_final = [drift_L["rho_A_at_alpha_eval"]["final"],
                   drift_P["rho_A_at_alpha_eval"]["final"]]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    x = np.arange(len(cfgs))
    width = 0.35
    axL.bar(x - width / 2, rho_W_init, width, label="init",
            color="#aac4e0", alpha=0.9)
    axL.bar(x + width / 2, rho_W_final, width, label="after 300 steps",
            color="#4C72B0", alpha=0.9)
    axL.axhline(1.0, color="gray", linestyle="--", linewidth=1.2)
    for i, (a, b) in enumerate(zip(rho_W_init, rho_W_final)):
        axL.text(i - width / 2, a + 0.03, f"{a:.2f}",
                 ha="center", va="bottom", fontsize=9)
        axL.text(i + width / 2, b + 0.03, f"{b:.2f}",
                 ha="center", va="bottom", fontsize=9)
    axL.set_xticks(x); axL.set_xticklabels(cfgs)
    axL.set_ylabel("ρ(W_rec)")
    axL.set_title("W_rec spectral radius drift under wake_phase")
    axL.legend(fontsize=9)

    axR.bar(x - width / 2, rho_A_init, width, label="init",
            color="#f0b8b8", alpha=0.9)
    axR.bar(x + width / 2, rho_A_final, width, label="after 300 steps",
            color="#C44E52", alpha=0.9)
    axR.axhline(1.0, color="gray", linestyle="--", linewidth=1.2,
                label="edge of chaos")
    for i, (a, b) in enumerate(zip(rho_A_init, rho_A_final)):
        axR.text(i - width / 2, a + 0.03, f"{a:.2f}",
                 ha="center", va="bottom", fontsize=9)
        axR.text(i + width / 2, b + 0.03, f"{b:.2f}",
                 ha="center", va="bottom", fontsize=9)
    axR.set_xticks(x); axR.set_xticklabels(cfgs)
    axR.set_ylabel("ρ(A) at operating α")
    axR.set_title("Linearised operator drift under wake_phase")
    axR.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "fig_drift_bars.png")


def _fig_linearisation(results: dict) -> None:
    L = results["linearisation_idle_after_training"]["legacy_config"]
    P = results["linearisation_idle_after_training"]["principled_config"]

    quantities = ["‖h‖ mean", "‖h‖ max", "|z| mean", "|z| max", "tanh'(z) mean"]
    vals_L = [L["h_norm_mean"], L["h_norm_max"], L["z_abs_mean"],
              L["z_abs_max"], L["tanh_prime_mean"]]
    vals_P = [P["h_norm_mean"], P["h_norm_max"], P["z_abs_mean"],
              P["z_abs_max"], P["tanh_prime_mean"]]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(quantities))
    width = 0.35
    ax.bar(x - width / 2, vals_L, width, label="legacy (s=0.1, α=0.8)",
           color="#4C72B0", alpha=0.9)
    ax.bar(x + width / 2, vals_P, width, label="principled (s*, α*)",
           color="#55A868", alpha=0.9)
    for i, (a, b) in enumerate(zip(vals_L, vals_P)):
        ax.text(i - width / 2, a + max(vals_L + vals_P) * 0.015,
                f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, b + max(vals_L + vals_P) * 0.015,
                f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(quantities)
    plt.setp(ax.get_xticklabels(), rotation=15)
    ax.set_ylabel("value (idle, 300 steps after training)")
    ax.set_title("Linearisation validity: is h ≈ 0 with tanh'(z) ≈ 1?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_linearisation.png")


# ── Public API ────────────────────────────────────────────────────────────────

def render(results: dict) -> None:
    print("\nRendering figures →", _figures_dir())
    _fig_scale_search(results)
    _fig_rho_sweep_at_sstar(results)
    _fig_drift_bars(results)
    _fig_linearisation(results)


def render_from_latest(name: str = "alpha_init",
                       repo_root: Optional[str] = None) -> None:
    payload = load_latest(name, repo_root=repo_root)
    render(payload["results"])


if __name__ == "__main__":
    render_from_latest()
