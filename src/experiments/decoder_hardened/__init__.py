"""
Industry-standard replication of the B2 decoder claim.

The workshop paper reports "1.000 held-out decoder accuracy on idle
states across 4 sinusoid frequencies." K=4 with hidden_dim=128 is at
ceiling: any distinguishable per-class geometry gives ~1.000. This
experiment raises the bar to K=20 (chance = 5%), n_seeds=30,
hidden_dim=256, and adds control decoders that stress the causal claim
"the idle-state geometry encodes task history."

What is tested, per seed:

    1. Main decoder (ridge head): K-way accuracy on per-task idle
       trajectories.
    2. Non-linear head control (MLP): same trajectories, MLP head.
       Distinguishes "linearly decodable" from "decodable at all."
    3. Shuffled-time control: timesteps are shuffled within each
       trajectory before the decoder runs. Destroys temporal structure
       while preserving marginals. If accuracy drops meaningfully, the
       fingerprint is trajectory-shape based (as claimed); if it stays
       high, the decoder was reading only per-timestep statistics.

At the population level (all seeds pooled), two more controls:

    4. Weight-norm decoder: does ‖W_rec_final‖ alone predict class?
       Networks all start from an IDENTICAL W_rec_init within a seed;
       divergence in ‖W_rec_final‖ across the K networks is a
       first-order confound to rule out.
    5. Mean-activation decoder: does the mean idle state alone predict
       class? Tests whether the fingerprint is one scalar per unit
       rather than a temporal geometry.

Statistics:
    Every per-seed accuracy comes with a bootstrap 95% CI over seeds.
    Population-level control decoders use leave-one-seed-out CV.
    Chance level = 1/K = 5%.

Non-goal: sleep phase. This experiment isolates "does the fingerprint
form at all" during wake training. The sleep-effect ablation and the
multi-cycle stream decoder are separate experiments (Phase B / C).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.harness import run_seeded
from src.metrics import (bootstrap_ci_mean, decoder_accuracy)
from src.model import LeakyRNN
from src.tasks import sample_specs
from src.train import run_idle, wake_phase, make_task


NAME = "decoder_hardened"
DESCRIPTION = ("Industry-standard replication of the idle-state decoder "
               "claim. K=20 sinusoid tasks, n=30 seeds, hidden_dim=256, "
               "with linear + MLP head, shuffled-time control, and "
               "population-level weight-norm and mean-activation controls.")
DETAILS = ("v1: LeakyRNN (hidden_dim=256, τ=5, target_sr=0.95, α=0.8). "
           "K=20 sine tasks with log-uniform frequency on [0.5, 20] Hz. "
           "Wake-only training (300 BPTT steps, lr=3e-3, gradient clip 1.0), "
           "no sleep phase. Idle trajectory 400 steps; last 300 used for "
           "decoder. Ridge λ=1.0, 5-fold CV within each seed. MLP: 64 hidden "
           "units, 200 epochs. n_seeds=30. Bootstrap 95% CI, 2000 resamples.")
VERSION = "v1"


# ── Canonical config ─────────────────────────────────────────────────────────

CONFIG: dict[str, Any] = {
    # Architecture
    "hidden_dim": 256,
    "input_dim":  16,
    "output_dim": 16,
    "tau":        5.0,
    "target_sr":  0.95,
    "alpha":      0.8,
    # Task suite
    "task_family":   "sine",
    "K":             20,
    "task_sample_seed": 0,
    # Wake training
    "wake_steps":  300,
    "wake_T":      200,
    "wake_batch":  4,
    "wake_lr":     3e-3,
    "wake_ach":    1.0,
    "grad_clip":   1.0,
    # Idle collection
    "idle_steps":  400,
    "trim":        300,
    # Decoder
    "ridge_lambda": 1.0,
    "n_splits":     5,
    "mlp_hidden":   64,
    "mlp_epochs":   200,
    "bootstrap_resamples": 2000,
    "bootstrap_ci":        0.95,
    # Seeds
    "n_seeds": 30,
    # Sleep phase INTENTIONALLY absent — this experiment isolates wake-only
    # fingerprint formation. Sleep effect is a separate hardened experiment.
}


# ── Per-seed body ────────────────────────────────────────────────────────────

def _train_one_task(seed: int, spec, device, config) -> tuple[np.ndarray, dict]:
    """Train a fresh LeakyRNN from `torch.manual_seed(seed)` on one task.

    Every task under one `seed` starts from the IDENTICAL initial weights
    (this matches the paper's setup: divergence is due to training data,
    not init randomness).

    Returns:
        idle_states: (idle_steps, hidden_dim) idle trajectory as float32 numpy.
        aux: {"W_rec_norm": float, "final_wake_loss": float}
    """
    torch.manual_seed(seed)
    m = LeakyRNN(config["input_dim"], config["hidden_dim"], config["output_dim"],
                 tau=config["tau"], alpha=config["alpha"],
                 target_sr=config["target_sr"]).to(device)

    inputs, targets = make_task(spec, T=config["wake_T"],
                                batch=config["wake_batch"],
                                input_dim=config["input_dim"],
                                seed=seed, device=device)
    losses = wake_phase(m, inputs, targets,
                        steps=config["wake_steps"],
                        lr=config["wake_lr"],
                        ach_gate=config["wake_ach"],
                        grad_clip=config["grad_clip"])

    states = run_idle(m, steps=config["idle_steps"], device=device)

    aux = {
        "W_rec_norm": float(m.W_rec.data.norm().item()),
        "W_rec_frobenius": float(m.W_rec.data.norm(p="fro").item()),
        "spectral_radius_final": float(m.spectral_radius()),
        "final_wake_loss": float(losses[-1]),
    }

    # Free model between tasks to keep memory flat
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return states.astype(np.float32), aux


def per_seed(seed: int, device, config: dict) -> dict:
    """Run the K-task fingerprint experiment for one seed."""
    # Task suite is fixed across seeds; that keeps class labels aligned
    # so the population-level controls can pool cleanly.
    specs = sample_specs(config["task_family"], config["K"],
                         seed=config["task_sample_seed"])

    try:
        from tqdm.auto import tqdm  # type: ignore
        spec_iter = tqdm(specs, unit="task", leave=False,
                          desc=f"seed {seed} tasks")
    except ImportError:
        spec_iter = specs

    idle_states: dict[str, np.ndarray] = {}
    task_aux: dict[str, dict] = {}
    for spec in spec_iter:
        states, aux = _train_one_task(seed, spec, device, config)
        idle_states[spec.label] = states
        task_aux[spec.label] = aux

    # Trim the last `trim` timesteps for the decoder
    trim = config["trim"]
    trimmed = {lbl: st[-trim:] for lbl, st in idle_states.items()}

    dec_ridge = decoder_accuracy(
        trimmed, n_splits=config["n_splits"],
        ridge_lambda=config["ridge_lambda"], trim=trim,
        head="ridge", shuffle_time=False, seed=seed)
    dec_ridge_shuffled = decoder_accuracy(
        trimmed, n_splits=config["n_splits"],
        ridge_lambda=config["ridge_lambda"], trim=trim,
        head="ridge", shuffle_time=True, seed=seed)
    dec_mlp = decoder_accuracy(
        trimmed, n_splits=config["n_splits"],
        ridge_lambda=config["ridge_lambda"], trim=trim,
        head="mlp", shuffle_time=False, seed=seed)

    # Mean states per class for the pooled control decoder
    mean_states = {lbl: st.mean(axis=0).tolist() for lbl, st in trimmed.items()}

    return {
        "labels":       [s.label for s in specs],
        "specs":        [s.to_dict() for s in specs],
        "decoder_ridge_main":     dec_ridge,
        "decoder_ridge_shuffled": dec_ridge_shuffled,
        "decoder_mlp_main":       dec_mlp,
        "task_aux":               task_aux,
        "mean_state_by_task":     mean_states,
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def _pool_leave_one_seed_out(features_by_seed: dict[int, dict[str, np.ndarray]],
                             labels: list[str],
                             ridge_lambda: float) -> dict:
    """Leave-one-seed-out K-way ridge decoder on per-network scalars/vectors.

    Each seed contributes ONE sample per class. We fit K-way ridge on
    the other seeds and predict this seed's K samples. Average
    per-sample accuracy is reported.

    Args:
        features_by_seed: {seed: {label: feature}}. Features are 1-D
            (scalar controls) or higher-D (vector controls).
        labels: canonical class order.
        ridge_lambda: L2 for ridge.

    Returns:
        dict with per-fold and pooled accuracy.
    """
    # Build (n_seeds*K, d) X and (n_seeds*K,) y arrays
    seeds_ordered = sorted(features_by_seed.keys())
    K = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}

    sample_feat: list[np.ndarray] = []
    sample_y: list[int] = []
    sample_seed: list[int] = []
    for s in seeds_ordered:
        d = features_by_seed[s]
        for l in labels:
            f = np.atleast_1d(np.asarray(d[l], dtype=np.float64))
            sample_feat.append(f)
            sample_y.append(label_idx[l])
            sample_seed.append(s)
    X = np.stack(sample_feat)
    y = np.array(sample_y)
    seed_arr = np.array(sample_seed)

    Y_oh = np.eye(K)[y]
    n_features = X.shape[1]

    fold_accs = []
    for held_out in seeds_ordered:
        test_mask = seed_arr == held_out
        train_mask = ~test_mask
        Xtr, Ytr = X[train_mask], Y_oh[train_mask]
        Xte, yte = X[test_mask], y[test_mask]
        XtX = Xtr.T @ Xtr + ridge_lambda * np.eye(n_features)
        W = np.linalg.solve(XtX, Xtr.T @ Ytr)
        preds = (Xte @ W).argmax(1)
        fold_accs.append(float((preds == yte).mean()))
    fold_accs = np.array(fold_accs)

    return {
        "mean":       float(fold_accs.mean()),
        "std":        float(fold_accs.std(ddof=1)) if len(fold_accs) > 1 else 0.0,
        "sem":        float(fold_accs.std(ddof=1) / np.sqrt(len(fold_accs)))
                       if len(fold_accs) > 1 else 0.0,
        "per_fold":   fold_accs.tolist(),
        "n_folds":    int(len(fold_accs)),
        "n_features": int(n_features),
        "ridge_lambda": float(ridge_lambda),
    }


def aggregate(per_seed_results: dict[int, dict], config: dict) -> dict:
    seeds = sorted(per_seed_results.keys())
    labels = per_seed_results[seeds[0]]["labels"]
    K = len(labels)
    chance = 1.0 / K

    # Extract per-seed accuracy arrays
    ridge_main_mean = np.array([per_seed_results[s]["decoder_ridge_main"]["mean"]
                                for s in seeds])
    ridge_shuf_mean = np.array([per_seed_results[s]["decoder_ridge_shuffled"]["mean"]
                                for s in seeds])
    mlp_main_mean = np.array([per_seed_results[s]["decoder_mlp_main"]["mean"]
                              for s in seeds])
    delta_shuf = ridge_main_mean - ridge_shuf_mean

    # Bootstrap CIs
    ci_ridge_main = bootstrap_ci_mean(
        ridge_main_mean, config["bootstrap_resamples"],
        config["bootstrap_ci"], seed=0)
    ci_ridge_shuf = bootstrap_ci_mean(
        ridge_shuf_mean, config["bootstrap_resamples"],
        config["bootstrap_ci"], seed=1)
    ci_mlp_main = bootstrap_ci_mean(
        mlp_main_mean, config["bootstrap_resamples"],
        config["bootstrap_ci"], seed=2)
    ci_delta_shuf = bootstrap_ci_mean(
        delta_shuf, config["bootstrap_resamples"],
        config["bootstrap_ci"], seed=3)

    # Population-level controls
    weight_norm_features = {
        s: {l: per_seed_results[s]["task_aux"][l]["W_rec_norm"]
            for l in labels}
        for s in seeds
    }
    mean_state_features = {
        s: {l: per_seed_results[s]["mean_state_by_task"][l]
            for l in labels}
        for s in seeds
    }
    ctrl_weight_norm = _pool_leave_one_seed_out(
        weight_norm_features, labels, config["ridge_lambda"])
    ctrl_mean_state = _pool_leave_one_seed_out(
        mean_state_features, labels, config["ridge_lambda"])

    # Verdict per test
    def _verdict(ci: dict, threshold: float, label: str) -> str:
        if ci["ci_lo"] > threshold:
            return f"{label}: ABOVE {threshold:.3f}"
        if ci["ci_hi"] < threshold:
            return f"{label}: BELOW {threshold:.3f}"
        return f"{label}: INCONCLUSIVE (CI crosses {threshold:.3f})"

    verdicts = [
        _verdict(ci_ridge_main, chance, "ridge_main vs chance"),
        _verdict(ci_ridge_shuf, chance, "ridge_shuffled vs chance"),
        _verdict(ci_mlp_main, chance, "mlp_main vs chance"),
        _verdict(ci_delta_shuf, 0.0, "ridge_main − ridge_shuffled > 0"),
    ]

    return {
        "K": K,
        "chance_level": chance,
        "n_seeds": len(seeds),
        "labels": labels,
        "task_specs": per_seed_results[seeds[0]]["specs"],
        "decoder_ridge_main": {
            "per_seed_mean": ridge_main_mean.tolist(),
            "bootstrap_ci": ci_ridge_main,
        },
        "decoder_ridge_shuffled": {
            "per_seed_mean": ridge_shuf_mean.tolist(),
            "bootstrap_ci": ci_ridge_shuf,
        },
        "decoder_mlp_main": {
            "per_seed_mean": mlp_main_mean.tolist(),
            "bootstrap_ci": ci_mlp_main,
        },
        "shuffled_vs_main_delta": {
            "per_seed_delta": delta_shuf.tolist(),
            "bootstrap_ci": ci_delta_shuf,
        },
        "population_controls": {
            "weight_norm_decoder": ctrl_weight_norm,
            "mean_state_decoder": ctrl_mean_state,
        },
        "verdicts": verdicts,
    }


# ── Entry point ──────────────────────────────────────────────────────────────

def run(device) -> dict:
    torch.set_default_dtype(torch.float32)
    seeds = list(range(CONFIG["n_seeds"]))
    final, _ = run_seeded(
        name=NAME,
        version=VERSION,
        seeds=seeds,
        per_seed_fn=per_seed,
        aggregate_fn=aggregate,
        config=CONFIG,
        device=device,
        description=DESCRIPTION,
        details=DETAILS,
        verbose=True,
    )

    # Human-readable summary
    a = final["aggregated"]
    print("\n" + "=" * 60)
    print(f"DECODER_HARDENED v{VERSION} summary")
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

    # Return the aggregated dict so run.py doesn't double-save
    return None
