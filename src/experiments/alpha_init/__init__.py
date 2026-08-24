"""
Numerical instantiation of the α-at-edge-of-chaos theorem (worksheet Part 6).

The theorem gives a unique α* ∈ (0, 1) with ρ(A(α*)) = 1 IF the hypotheses
(H1) ρ(W_rec) < 1 and (H2) ρ(A(0)) > 1 hold. This experiment applies the
theorem to the actual LeakyRNN init at seed=42 and reports what happens.

Structure of the experiment:

  (1)  Report the situation at the codebase's current scale (s=0.1 on
       both W_in and W_out). If (H2) fails, this documents why.
  (2)  Sweep joint scale s to find s* where (H2) and empirical
       monotonicity of α → ρ(A(α)) both hold.
  (3)  Compute α* at s* and report every quantity in Assumption (E).
  (4)  Compare: ρ(A) at (s=0.1, α=0.8) vs (s*, α*). Magnitude ratio too.
  (5)  Post-training drift: run wake_phase on freq=3, then re-evaluate
       ρ(A(α*)) on the trained weights. Report drift.
  (6)  Linearisation validity: measure ‖h‖ and tanh'(z) during idle,
       so a reviewer can judge whether the h≈0 linearisation applies
       during operation.

Figures live in `figures.py`. Finding: findings/week<NN>_<YYYY>/…
"""

from __future__ import annotations

import numpy as np
import torch

from src.alpha import (
    compute_alpha,
    find_valid_scale,
    rho_A,
    spectral_radius,
    report_to_dict,
)
from src.model import LeakyRNN
from src.train import make_sine_task, wake_phase


NAME = "alpha_init"
DESCRIPTION = ("Numerical instantiation of the α-at-edge-of-chaos theorem "
               "on the actual LeakyRNN init (seed=42). Finds the smallest "
               "joint W_in/W_out scale that satisfies the theorem's "
               "hypotheses; derives α* there; measures post-training drift "
               "and linearisation validity.")
DETAILS = ("v1: seed=42, hidden_dim=128, input_dim=16, output_dim=16, τ=5, "
           "target_sr=0.95. Codebase legacy config is (s=0.1, α=0.8). "
           "Bisection tolerance 1e-8. Drift uses wake_phase(freq=3, steps=300).")
VERSION = "v1"

# Config
INPUT_DIM = 16
HIDDEN_DIM = 128
OUTPUT_DIM = 16
TAU = 5.0
TARGET_SR = 0.95
SEED = 42
LEGACY_SCALE = 0.1           # what LeakyRNN.__init__ uses today
LEGACY_ALPHA = 0.8           # what benchmark.py / baselines.py use

DRIFT_TRAIN_FREQ = 3.0
DRIFT_TRAIN_STEPS = 300
IDLE_STEPS = 300


# ── Torch → numpy weight extraction ───────────────────────────────────────────

def _extract_weights_torch(model: LeakyRNN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W_rec = model.W_rec.detach().cpu().numpy().astype(np.float64)
    W_in = model.W_in.detach().cpu().numpy().astype(np.float64)
    W_out = model.W_out.detach().cpu().numpy().astype(np.float64)
    return W_rec, W_in, W_out


def _fresh_leakyrnn(device, alpha: float = LEGACY_ALPHA) -> LeakyRNN:
    """Reproducible LeakyRNN at seed=SEED. `alpha` is a runtime blend, not
    a weight scale, so it doesn't affect the extracted weights."""
    torch.manual_seed(SEED)
    return LeakyRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
                    tau=TAU, alpha=alpha, target_sr=TARGET_SR).to(device)


def _rescale_model_(model: LeakyRNN, s: float) -> None:
    """Rescale W_in (buffer) and W_out (parameter) by s/LEGACY_SCALE in-place.
    Everything else is untouched. Preserves the seed=42 random *pattern* but
    changes the *scale*."""
    factor = s / LEGACY_SCALE
    with torch.no_grad():
        model.W_in.mul_(factor)
        model.W_out.mul_(factor)


# ── Post-training drift + linearisation ───────────────────────────────────────

def _measure_h_and_z(model: LeakyRNN, device, steps: int) -> dict:
    """
    Idle-run the model and measure the linearisation-relevant quantities:
        ‖h(t)‖, |z(t)|, and tanh'(z(t)) per unit
    where z(t) = fb·W_inᵀ + h·W_recᵀ + b (argument going into tanh).

    Reported: mean and max of each across `steps` idle steps.
    """
    model.eval()
    h = model.init_hidden(1, device)
    h_norms, z_abs_mean, z_abs_max, tanh_prime_mean = [], [], [], []

    with torch.no_grad():
        for _ in range(steps):
            fb = h @ model.W_out.T
            inp = fb @ model.W_in.T
            z = inp + h @ model.W_rec.T + model.bias
            tp = 1.0 - torch.tanh(z) ** 2

            h_norms.append(float(h.norm(dim=-1).mean()))
            z_abs_mean.append(float(z.abs().mean()))
            z_abs_max.append(float(z.abs().max()))
            tanh_prime_mean.append(float(tp.mean()))

            h = (1.0 - model.leak) * h + model.leak * torch.tanh(z)

    return {
        "steps": steps,
        "h_norm_mean": float(np.mean(h_norms)),
        "h_norm_max": float(np.max(h_norms)),
        "z_abs_mean": float(np.mean(z_abs_mean)),
        "z_abs_max": float(np.max(z_abs_max)),
        "tanh_prime_mean": float(np.mean(tanh_prime_mean)),
        "tanh_prime_min": float(np.min(tanh_prime_mean)),
    }


def _train_and_evaluate(model: LeakyRNN, device, alpha_eval: float,
                        tau: float) -> dict:
    """
    Train `model` for DRIFT_TRAIN_STEPS on a freq=DRIFT_TRAIN_FREQ sine.
    Extract weights before and after. Report drift of ρ(A(α_eval)) and,
    when the theorem still applies, α*_final.
    """
    W_rec_i, W_in, W_out_i = _extract_weights_torch(model)
    rho_A_init_at_alpha_eval = rho_A(W_rec_i, W_in, W_out_i, tau, alpha_eval)

    inputs, targets = make_sine_task(freq=DRIFT_TRAIN_FREQ, device=device)
    losses = wake_phase(model, inputs, targets, steps=DRIFT_TRAIN_STEPS,
                        lr=3e-3, ach_gate=1.0)

    W_rec_f, _, W_out_f = _extract_weights_torch(model)
    rho_W_rec_final = spectral_radius(W_rec_f)
    delta_W_rec = float(np.linalg.norm(W_rec_f - W_rec_i, ord="fro"))
    delta_W_out = float(np.linalg.norm(W_out_f - W_out_i, ord="fro"))

    rho_A_final_at_alpha_eval = rho_A(W_rec_f, W_in, W_out_f, tau, alpha_eval)
    rho_A0_final = rho_A(W_rec_f, W_in, W_out_f, tau, 0.0)
    rho_A1_final = rho_A(W_rec_f, W_in, W_out_f, tau, 1.0)

    # α*_final: does the theorem still apply after training?
    try:
        rep_f = compute_alpha(W_rec_f, W_in, W_out_f, tau=tau)
        drift = {
            "theorem_still_applicable": True,
            "alpha_star_final": rep_f.alpha_star,
            "rho_A_at_alpha_star_final": rep_f.rho_A_at_alpha_star,
            "E_ii_min_inner_product_real_final":
                rep_f.E_diagnostics["E_ii_min_inner_product_real"],
        }
    except ValueError as err:
        drift = {
            "theorem_still_applicable": False,
            "failure_reason": str(err),
            "alpha_star_final": None,
        }

    return {
        "training_loss": {"initial": float(losses[0]), "final": float(losses[-1])},
        "rho_W_rec": {"init": float(spectral_radius(W_rec_i)),
                      "final": float(rho_W_rec_final)},
        "rho_A_at_alpha_eval": {"init": float(rho_A_init_at_alpha_eval),
                                "final": float(rho_A_final_at_alpha_eval),
                                "alpha_eval": float(alpha_eval)},
        "rho_A_final_at_0": float(rho_A0_final),
        "rho_A_final_at_1": float(rho_A1_final),
        "delta_W_rec_frobenius": delta_W_rec,
        "delta_W_out_frobenius": delta_W_out,
        **drift,
    }


# ── Main experiment ───────────────────────────────────────────────────────────

def run(device) -> dict:
    torch.set_default_dtype(torch.float32)
    print(f"\nα-init experiment: n={HIDDEN_DIM}, d={INPUT_DIM}, τ={TAU}, seed={SEED}")

    # ── (1) Situation at the current codebase scale ─────────────────────────
    print(f"\n[1/6] Codebase default scale (s={LEGACY_SCALE}, α={LEGACY_ALPHA})...")
    m_legacy = _fresh_leakyrnn(device)                 # s=0.1 by construction
    W_rec_L, W_in_L, W_out_L = _extract_weights_torch(m_legacy)

    rho_W_L = spectral_radius(W_rec_L)
    rho_A0_L = rho_A(W_rec_L, W_in_L, W_out_L, TAU, 0.0)
    rho_A_at_legacy_alpha = rho_A(W_rec_L, W_in_L, W_out_L, TAU, LEGACY_ALPHA)
    theorem_applies_at_legacy = rho_W_L < 1.0 and rho_A0_L > 1.0

    legacy_report: dict = {
        "s": LEGACY_SCALE,
        "alpha": LEGACY_ALPHA,
        "rho_W_rec": float(rho_W_L),
        "rho_A_at_0": float(rho_A0_L),
        "rho_A_at_legacy_alpha": float(rho_A_at_legacy_alpha),
        "H1_holds": bool(rho_W_L < 1.0),
        "H2_holds": bool(rho_A0_L > 1.0),
        "theorem_applies": theorem_applies_at_legacy,
    }

    print(f"       ρ(W_rec)              = {rho_W_L:.6f}  (H1 holds: {rho_W_L < 1.0})")
    print(f"       ρ(A(0))               = {rho_A0_L:.6f}  (H2 holds: {rho_A0_L > 1.0})")
    print(f"       ρ(A(α={LEGACY_ALPHA}))         = {rho_A_at_legacy_alpha:.6f}")
    if not theorem_applies_at_legacy:
        print("       → Theorem does NOT apply at s=0.1. No α* exists in (0,1).")

    # If (H2) holds at the legacy scale, we can still compute α* there for
    # completeness. If not, this branch simply notes the failure.
    if theorem_applies_at_legacy:
        legacy_r = compute_alpha(W_rec_L, W_in_L, W_out_L, tau=TAU)
        legacy_report["alpha_star_at_legacy_scale"] = legacy_r.alpha_star
        legacy_report["rho_A_at_legacy_alpha_star"] = legacy_r.rho_A_at_alpha_star

    # ── (2) Scale search ────────────────────────────────────────────────────
    print("\n[2/6] Searching for the smallest scale s where the theorem applies...")
    # We reconstruct unit-scale weights: divide out the LEGACY_SCALE that the
    # LeakyRNN constructor already applied. This preserves the seed=42
    # *random pattern*; only the scale changes.
    W_in_unit = W_in_L / LEGACY_SCALE
    W_out_unit = W_out_L / LEGACY_SCALE

    scale_search = find_valid_scale(W_rec_L, W_in_unit, W_out_unit,
                                    tau=TAU, tol=1e-4)
    s_star = scale_search["s_star"]
    if s_star is None:
        raise RuntimeError("no valid scale found in [0.05, 2.0]; "
                           "widen the search range")
    print(f"       (H2) alone first holds at    s ≈ {scale_search['transition_H2_s']:.4f}")
    print(f"       monotonicity first holds at  s ≈ {scale_search['transition_mono_s']:.4f}")
    print(f"       both hold from               s ≈ {s_star:.4f} onward")

    # ── (3) Theorem verification at s* ──────────────────────────────────────
    print(f"\n[3/6] Theorem verification at s* = {s_star:.4f}...")
    W_in_star = s_star * W_in_unit
    W_out_star = s_star * W_out_unit
    report_star = compute_alpha(W_rec_L, W_in_star, W_out_star, tau=TAU)
    print(f"       ρ(W_rec)              = {report_star.H1_rho_W_rec:.6f}  (H1: {report_star.H1_holds})")
    print(f"       ρ(A(0))               = {report_star.H2_rho_A_at_0:.6f}  (H2: {report_star.H2_holds})")
    print(f"       α*                    = {report_star.alpha_star:.8f}")
    print(f"       ρ(A(α*))              = {report_star.rho_A_at_alpha_star:.8f}")
    print(f"       (E.i) min gap         = "
          f"{report_star.E_diagnostics['E_i_min_eigenvalue_gap']:.4f}")
    print(f"       (E.ii) min inner      = "
          f"{report_star.E_diagnostics['E_ii_min_inner_product_real']:.4f}  "
          f"(>0 ⇒ ρ(A(α)) strictly decreasing)")
    print(f"       monotonicity          = {report_star.monotonicity_sweep['monotone_non_increasing']} "
          f"(worst gap {report_star.monotonicity_sweep['worst_gap']:.2e})")

    # ── (4) Magnitude ratio external:feedback ───────────────────────────────
    print(f"\n[4/6] Magnitude ratio (external : feedback)...")

    def _magnitude_ratio(alpha: float, W_in: np.ndarray, W_out: np.ndarray) -> float:
        # ‖x · W_inᵀ‖ scales as α · rms_x · ‖W_inᵀ‖  ;  rms_x = 1/√2 for a sine
        # ‖fb · W_inᵀ‖ scales as (1-α) · ‖h‖ · ‖W_outᵀ · W_inᵀ‖
        # Use ‖h‖ = 0.3 as a stand-in.
        num = alpha * (1.0 / np.sqrt(2.0)) * float(np.linalg.norm(W_in.T))
        den = (1.0 - alpha) * 0.3 * float(np.linalg.norm(W_out.T @ W_in.T))
        return num / max(den, 1e-30)

    ratio_legacy = _magnitude_ratio(LEGACY_ALPHA, W_in_L, W_out_L)
    ratio_star = _magnitude_ratio(report_star.alpha_star, W_in_star, W_out_star)
    print(f"       at (s=0.1, α=0.8)  → {ratio_legacy:.2f} : 1")
    print(f"       at (s*, α*)        → {ratio_star:.2f} : 1")

    # ── (5) Post-training drift, both configurations ────────────────────────
    print(f"\n[5/6] Post-training drift (wake_phase freq={DRIFT_TRAIN_FREQ}, "
          f"{DRIFT_TRAIN_STEPS} steps)...")

    print("       (5a) legacy config (s=0.1, α=0.8)")
    m_L = _fresh_leakyrnn(device, alpha=LEGACY_ALPHA)
    drift_legacy = _train_and_evaluate(m_L, device, alpha_eval=LEGACY_ALPHA, tau=TAU)
    lin_legacy = _measure_h_and_z(m_L, device, IDLE_STEPS)
    print(f"           ρ(W_rec) {drift_legacy['rho_W_rec']['init']:.4f} → "
          f"{drift_legacy['rho_W_rec']['final']:.4f}")
    print(f"           ρ(A(0.8)) {drift_legacy['rho_A_at_alpha_eval']['init']:.4f} → "
          f"{drift_legacy['rho_A_at_alpha_eval']['final']:.4f}")

    print(f"       (5b) principled config (s={s_star:.4f}, α={report_star.alpha_star:.4f})")
    m_S = _fresh_leakyrnn(device, alpha=report_star.alpha_star)
    _rescale_model_(m_S, s_star)
    drift_star = _train_and_evaluate(m_S, device, alpha_eval=report_star.alpha_star,
                                     tau=TAU)
    lin_star = _measure_h_and_z(m_S, device, IDLE_STEPS)
    print(f"           ρ(W_rec) {drift_star['rho_W_rec']['init']:.4f} → "
          f"{drift_star['rho_W_rec']['final']:.4f}")
    print(f"           ρ(A(α*)) {drift_star['rho_A_at_alpha_eval']['init']:.4f} → "
          f"{drift_star['rho_A_at_alpha_eval']['final']:.4f}")

    # ── (6) Linearisation validity (already measured inline in step 5) ─────
    print(f"\n[6/6] Linearisation validity (idle {IDLE_STEPS} steps after training)")
    print(f"       legacy:      ‖h‖ mean={lin_legacy['h_norm_mean']:.3f}, "
          f"tanh'(z) mean={lin_legacy['tanh_prime_mean']:.3f}")
    print(f"       principled:  ‖h‖ mean={lin_star['h_norm_mean']:.3f}, "
          f"tanh'(z) mean={lin_star['tanh_prime_mean']:.3f}")

    # ── Assemble result ─────────────────────────────────────────────────────
    out = {
        "config": {
            "hidden_dim": HIDDEN_DIM, "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM, "tau": TAU, "target_sr": TARGET_SR,
            "seed": SEED, "device": str(device),
            "legacy_scale": LEGACY_SCALE, "legacy_alpha": LEGACY_ALPHA,
        },
        "legacy_scale_diagnostics": legacy_report,
        "scale_search": scale_search,
        "principled_alpha": {
            "s_star": float(s_star),
            **report_to_dict(report_star),
        },
        "magnitude_ratio_external_over_feedback": {
            "at_legacy_config": float(ratio_legacy),
            "at_principled_config": float(ratio_star),
            "assumptions": {
                "external_rms": 1.0 / np.sqrt(2.0),
                "hidden_state_norm_estimate": 0.3,
                "note": "operator-norm-based scale estimate; empirical values "
                        "measured downstream in the 3-point sanity sweep.",
            },
        },
        "drift": {
            "legacy_config": drift_legacy,
            "principled_config": drift_star,
        },
        "linearisation_idle_after_training": {
            "legacy_config": lin_legacy,
            "principled_config": lin_star,
        },
    }

    try:
        from src.experiments.alpha_init import figures as fig_mod
        fig_mod.render(out)
    except Exception as e:
        print(f"\n(skipping figures: {e})")

    return out
