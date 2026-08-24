"""
Task-family replication of `decoder_hardened` on Lorenz attractors.

Same protocol as `decoder_hardened` (K=20 tasks, n=30 seeds, wake-only,
ridge / MLP / shuffled-time / weight-norm / mean-state controls, bootstrap
95% CI over seeds), but the task family is `lorenz` (chaotic 3D attractor,
parameterised by ρ) instead of `sine`.

Lorenz is deterministic but chaotic. The 20 "tasks" are 20 different ρ
values on [15, 40]. Tests whether the phenomenon survives when the
target is aperiodic and long-range unpredictable.

Note. Lorenz task generation involves ODE integration and is slower than
the other families. Per-seed wallclock will be higher.
"""

from __future__ import annotations

import torch

from src.experiments.decoder_hardened import (
    per_seed, aggregate, CONFIG as _BASE_CONFIG,
)
from src.harness import run_seeded


NAME = "decoder_hardened_lorenz"
DESCRIPTION = ("Task-family replication of decoder_hardened on Lorenz "
               "trajectories. K=20 ρ values (uniform on [15, 40]), "
               "n=30 seeds, same controls.")
DETAILS = ("v1: identical protocol to decoder_hardened v1 except "
           "task_family='lorenz'. ρ sampled uniformly on [15, 40]. "
           "Lorenz trajectories generated via RK4 integration (see "
           "src.tasks._lorenz). Slower than other families due to "
           "the ODE integration.")
VERSION = "v1"


CONFIG = dict(_BASE_CONFIG)
CONFIG["task_family"] = "lorenz"


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
    print(f"DECODER_HARDENED_LORENZ v{VERSION} summary")
    print("=" * 60)
    print(f"K = {a['K']}  (chance = {a['chance_level']:.3f})")
    print(f"n_seeds = {a['n_seeds']}")
    for key in ["decoder_ridge_main", "decoder_ridge_shuffled",
                "decoder_mlp_main"]:
        ci = a[key]["bootstrap_ci"]
        print(f"  {key:28s} = {ci['mean']:.3f}  "
              f"95% CI [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]")
    d = a["shuffled_vs_main_delta"]["bootstrap_ci"]
    print(f"  main − shuffled              = {d['mean']:+.3f}  "
          f"95% CI [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]")
    print("\n  Population controls (leave-one-seed-out):")
    for key in ["weight_norm_decoder", "mean_state_decoder"]:
        c = a["population_controls"][key]
        print(f"    {key:22s} mean = {c['mean']:.3f}  sem = {c['sem']:.3f}")
    print("\n  Verdicts:")
    for v in a["verdicts"]:
        print(f"    - {v}")
    return None
