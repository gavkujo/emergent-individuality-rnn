"""
Generate all figures from results/benchmark_results.json.
Run after benchmark.py. Saves PNGs to figures/.

  fig1_idle_richness.png   — GRU vs LeakyRNN idle richness
  fig2_decoder_acc.png     — Decoder accuracy: single task vs 4 streams
  fig3_accumulation.png    — Subspace angle over 5 wake-sleep cycles
  fig4_path_dependence.png — Subspace angle vs stream length
  fig5_sleep_effect.png    — With vs without sleep: angle and L2
  fig6_order_effect.png    — Order effect: same-reversed vs different tasks
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS_PATH = os.path.join(ROOT, "results", "benchmark_results.json")
FIGURES_DIR  = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

with open(RESULTS_PATH) as f:
    R = json.load(f)


def save(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"✓ {name}")


# Fig 1: Idle richness
fig, ax = plt.subplots(figsize=(7, 4))
gru_r = R["b1_idle_richness"]["gru_richness"]
leaky = R["b1_idle_richness"]["leaky"]
taus, srs = [2.0, 5.0, 10.0], [0.9, 0.95, 1.05]
x, width = np.arange(len(taus)), 0.22
colors = ["#4C72B0", "#55A868", "#C44E52"]
for j, sr in enumerate(srs):
    vals = [leaky[f"tau{tau}_sr{sr}"]["richness"] for tau in taus]
    ax.bar(x + j*width, vals, width, label=f"SR={sr}", color=colors[j], alpha=0.85)
ax.axhline(gru_r, color="black", linestyle="--", linewidth=1.5,
           label=f"GRU baseline ({gru_r:.6f})")
ax.set_xticks(x + width); ax.set_xticklabels([f"τ={t}" for t in taus])
ax.set_ylabel("Idle richness (std of hidden state over time)")
ax.set_title("Fig 1 — Idle Dynamics Richness: GRU vs LeakyRNN")
ax.legend(fontsize=9); plt.tight_layout(); save("fig1_idle_richness.png")


# Fig 2: Decoder accuracy
fig, ax = plt.subplots(figsize=(
5, 4))
accs = [R["b2_decoder"]["single_task_acc"], R["b2_decoder"]["four_stream_acc"]]
bars = ax.bar(["Single task\n(1 experience)", "4 streams\n(5 cycles each)"],
              accs, color=["#4C72B0", "#55A868"], width=0.4, alpha=0.85)
ax.axhline(0.25, color="gray", linestyle="--", linewidth=1.2, label="Chance (0.25)")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, acc + 0.01, f"{acc:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.12); ax.set_ylabel("Linear decoder accuracy")
ax.set_title("Fig 2 — Experiential History Decodability")
ax.legend(fontsize=9); plt.tight_layout(); save("fig2_decoder_acc.png")


# Fig 3: Divergence accumulation
fig, ax1 = plt.subplots(figsize=(6, 4))
cycles = [d["cycle"] for d in R["b3_accumulation"]]
angles = [d["subspace_angle"] for d in R["b3_accumulation"]]
l2s    = [d["l2"] for d in R["b3_accumulation"]]
ax1.plot(cycles, angles, "o-", color="#4C72B0", linewidth=2, markersize=7, label="Subspace angle (°)")
ax1.set_xlabel("Wake-sleep cycle"); ax1.set_ylabel("Subspace angle (°)", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax2 = ax1.twinx()
ax2.plot(cycles, l2s, "s--", color="#C44E52", linewidth=1.5, markersize=6, label="L2 distance")
ax2.set_ylabel("L2 distance", color="#C44E52"); ax2.tick_params(axis="y", labelcolor="#C44E52")
ax1.set_title("Fig 3 — Divergence Accumulation Across Wake-Sleep Cycles")
l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, fontsize=9, loc="lower right")
plt.tight_layout(); save("fig3_accumulation.png")


# Fig 4: Path-dependence
fig, ax = plt.subplots(figsize=(6, 4))
lengths  = [d["length"] for d in R["b4_path_dependence"]]
p_angles = [d["angle"]  for d in R["b4_path_dependence"]]
ax.plot(lengths, p_angles, "o-", color="#55A868", linewidth=2, markersize=7)
ax.fill_between(lengths, [a-2 for a in p_angles], [a+2 for a in p_angles],
                alpha=0.15, color="#55A868")
ax.set_xlabel("Stream length (wake-sleep cycles)")
ax.set_ylabel("Subspace angle (°) — forward vs reversed")
ax.set_title("Fig 4 — Path-Dependence: Order of Experience Matters")
ax.set_xticks(lengths)
ax.annotate(f"{p_angles[0]:.1f}°", (lengths[0], p_angles[0]),
            textcoords="offset points", xytext=(8, 4), fontsize=9)
ax.annotate(f"{p_angles[-1]:.1f}°", (lengths[-1], p_angles[-1]),
            textcoords="offset points", xytext=(-20, 6), fontsize=9)
plt.tight_layout(); save("fig4_path_dependence.png")


# Fig 5: Sleep effect
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
conditions = ["Without sleep", "With sleep"]
angle_vals = [R["b5_sleep_effect"]["without_sleep"]["angle"],
              R["b5_sleep_effect"]["with_sleep"]["angle"]]
l2_vals    = [R["b5_sleep_effect"]["without_sleep"]["l2"],
              R["b5_sleep_effect"]["with_sleep"]["l2"]]
for ax, vals, ylabel, title, clr in zip(
    axes, [angle_vals, l2_vals],
    ["Subspace angle (°)", "L2 distance"],
    ["Subspace Angle", "L2 Distance"],
    [["#aac4e0", "#4C72B0"], ["#f0b8b8", "#C44E52"]],
):
    bars = ax.bar(conditions, vals, color=clr, width=0.4, alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, max(vals)*1.2)
fig.suptitle("Fig 5 — Effect of Sleep on Representational Divergence", fontsize=12)
plt.tight_layout(); save("fig5_sleep_effect.png")


# Fig 6: Order effect
fig, ax = plt.subplots(figsize=(5, 4))
oe   = R["b6_order_effect"]
vals = [oe["same_tasks_reversed"]["angle"], oe["different_tasks"]["angle"]]
bars = ax.bar(["Same tasks,\nreversed order\n(A vs D)", "Completely\ndifferent tasks\n(A vs B)"],
              vals, color=["#8172B2", "#C44E52"], width=0.4, alpha=0.85)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
            f"{v:.1f}°", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Subspace angle (°)")
ax.set_title(f"Fig 6 — Order Effect\nSame-reversed = {oe['order_fraction']:.0%} of different-task divergence")
ax.set_ylim(0, max(vals)*1.25)
plt.tight_layout(); save("fig6_order_effect.png")

print(f"\nAll figures saved to {FIGURES_DIR}")
