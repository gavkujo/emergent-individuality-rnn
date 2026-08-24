"""
α derivation for the self-feedback leaky-integrator RNN.

Given W_rec, W_in, W_out, τ, choose α ∈ [0, 1] so that the linearised
one-step recurrent operator sits at the edge of chaos:

    A(α)  =  (1 − 1/τ)·I  +  (1/τ)·[ W_recᵀ + (1 − α)·W_outᵀ·W_inᵀ ]
    M(α)  =  W_rec + (1 − α) · W_in · W_out                    (auxiliary)
    ρ(A(α*)) = 1                                               (criterion)

Theorem, hypotheses, and proof are in
`supporting/working/alpha_derivation_proof_yours.md`. This module contains
the computational machinery only.

Everything here is pure numpy. Callers convert torch tensors to numpy at
the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np


# ── Core linear-algebra primitives ────────────────────────────────────────────

def M_of_alpha(W_rec: np.ndarray, W_in: np.ndarray, W_out: np.ndarray,
               alpha: float) -> np.ndarray:
    """M(α) = W_rec + (1-α) · W_in · W_out.

    Shapes: W_rec (n,n), W_in (n,d), W_out (d,n) → returns (n,n).
    """
    return W_rec + (1.0 - alpha) * (W_in @ W_out)


def A_of_alpha(W_rec: np.ndarray, W_in: np.ndarray, W_out: np.ndarray,
               tau: float, alpha: float) -> np.ndarray:
    """A(α) = (1-1/τ)·I + (1/τ)·M(α)ᵀ.

    This is the linearised one-step recurrent operator acting on the hidden
    state (rows-are-vectors convention matching the code in model.py).
    """
    n = W_rec.shape[0]
    M = M_of_alpha(W_rec, W_in, W_out, alpha)
    return (1.0 - 1.0 / tau) * np.eye(n) + (1.0 / tau) * M.T


def spectral_radius(matrix: np.ndarray) -> float:
    """ρ(M) = max_i |λᵢ(M)|. Standard eigenvalue solve."""
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def rho_A(W_rec: np.ndarray, W_in: np.ndarray, W_out: np.ndarray,
          tau: float, alpha: float) -> float:
    """ρ(A(α)). Convenience wrapper."""
    return spectral_radius(A_of_alpha(W_rec, W_in, W_out, tau, alpha))


# ── Root-finding (bisection; monotone in α under Assumption E) ────────────────

def _bisect(f: Callable[[float], float], lo: float, hi: float,
            xtol: float = 1e-8, max_iter: int = 200) -> tuple[float, dict]:
    """
    Bisection with sign-change check. Assumes f(lo) and f(hi) have opposite
    signs. Returns (root, diagnostics).
    """
    flo = f(lo)
    fhi = f(hi)
    if flo == 0.0:
        return lo, {"iterations": 0, "converged": True, "f_root": flo}
    if fhi == 0.0:
        return hi, {"iterations": 0, "converged": True, "f_root": fhi}
    if flo * fhi > 0.0:
        raise ValueError(
            f"bisection failed: f(lo)={flo:.6f} and f(hi)={fhi:.6f} have same sign"
        )

    a, b = lo, hi
    fa = flo
    for it in range(1, max_iter + 1):
        mid = 0.5 * (a + b)
        fmid = f(mid)
        if abs(fmid) < xtol or (b - a) < xtol:
            return mid, {"iterations": it, "converged": True, "f_root": float(fmid)}
        if fa * fmid < 0.0:
            b = mid
        else:
            a, fa = mid, fmid
    return 0.5 * (a + b), {"iterations": max_iter, "converged": False,
                           "f_root": float(fmid)}


# ── Perturbation-theory diagnostics for Assumption (E) ────────────────────────

def dominant_eigenpair(matrix: np.ndarray) -> tuple[complex, np.ndarray, np.ndarray, float]:
    """
    Return (λ*, v, u, gap) where:
      λ* : dominant eigenvalue (largest |·|)
      v  : right eigenvector, unit norm
      u  : left eigenvector, normalised so uᵀ·v = 1 (biorthogonal to v)
      gap: |λ*| − |λ_second| (the eigenvalue-gap; > 0 iff simple-dominant)

    The left eigenvector is the right eigenvector of the transpose,
    corresponding to the same eigenvalue.
    """
    lam_r, V_r = np.linalg.eig(matrix)
    lam_l, V_l = np.linalg.eig(matrix.T)

    # Rank eigenvalues of `matrix` by |·|; pick the top
    order_r = np.argsort(-np.abs(lam_r))
    idx_top = order_r[0]
    idx_second = order_r[1] if len(order_r) > 1 else order_r[0]
    lam_star = lam_r[idx_top]
    v = V_r[:, idx_top]

    # Match the corresponding left eigenvector by minimising |lam_l - lam_star|
    match = np.argmin(np.abs(lam_l - lam_star))
    u = V_l[:, match]

    # Normalise: unit right eigenvector; left rescaled so uᵀv = 1.
    v = v / np.linalg.norm(v)
    denom = u @ v
    if np.abs(denom) < 1e-14:
        # numerically degenerate; fall back to a large-magnitude but
        # unnormalised left vector to avoid NaNs downstream
        u = u / np.linalg.norm(u)
    else:
        u = u / denom

    gap = float(np.abs(lam_star) - np.abs(lam_r[idx_second]))
    return complex(lam_star), v, u, gap


def perturbation_derivative(W_rec: np.ndarray, W_in: np.ndarray,
                            W_out: np.ndarray, tau: float,
                            alpha: float) -> dict:
    """
    Analytic derivative of the dominant eigenvalue of A(α) with respect to α.

    Using Part 4.4 of the worksheet:
        dλ_M*/dα  = − uᵀ · (W_in · W_out) · v         (perturbation formula)
        dλ_A*/dα  = (1/τ) · dλ_M*/dα                  (chain rule via (4.2.1))
        d|λ_A*|²/dα = 2·Re( conj(λ_A*) · dλ_A*/dα )   (envelope)

    Assumption (E.ii) is  Re(conj(λ_A*) · uᵀ · (W_in · W_out) · v) > 0,
    which is equivalent to  d|λ_A*|²/dα < 0 at this α.
    """
    A = A_of_alpha(W_rec, W_in, W_out, tau, alpha)
    lam_A, v_A, u_A, gap_A = dominant_eigenpair(A)

    # Perturbation from M's side (A eigenvalue derivatives come via (4.2.1)).
    # Note: the eigenvectors of A(α) = c·I + (1/τ)·M(α)ᵀ are the same as those
    # of M(α)ᵀ. The left eigenvector of A is the left eigenvector of M(α)ᵀ,
    # which is the right eigenvector of M(α). So the biorthogonal pair (u_M, v_M)
    # for M(α) is (v_A, u_A) up to conjugation of the shift.
    # Simpler: compute the derivative directly from A.

    perturbation = W_in @ W_out  # (n, n), independent of α
    # For A(α) the perturbation is -(1/τ)·(perturbation)ᵀ:
    #   dA/dα = -(1/τ) · (W_inᵀ)(W_outᵀ)  — actually apply the transpose.
    dA_dalpha = -(1.0 / tau) * (perturbation.T)  # note the transpose
    dlambda_dalpha = complex(u_A @ dA_dalpha @ v_A)

    d_abs2_dalpha = 2.0 * float(np.real(np.conj(lam_A) * dlambda_dalpha))
    inner = float(np.real(np.conj(lam_A) * (u_A @ perturbation.T @ v_A)))

    return {
        "alpha": float(alpha),
        "lambda_A_star": {"real": float(lam_A.real), "imag": float(lam_A.imag),
                          "abs": float(np.abs(lam_A))},
        "eigenvalue_gap_A": gap_A,
        "dlambda_dalpha": {"real": float(dlambda_dalpha.real),
                           "imag": float(dlambda_dalpha.imag)},
        "d_abs_lambda2_dalpha": d_abs2_dalpha,           # < 0 iff |λ_A| decreasing
        "E_ii_inner_product_real": inner,                # > 0 iff (E.ii) holds
        "E_i_simple_dominant": gap_A > 1e-8,
    }


# ── Verification pipeline ─────────────────────────────────────────────────────

@dataclass
class VerificationReport:
    H1_rho_W_rec: float
    H1_holds: bool
    H2_rho_A_at_0: float
    H2_holds: bool
    monotonicity_sweep: dict         # {alphas, rho_A, worst_gap, monotone}
    E_diagnostics: dict              # {alpha_star_value?, eigenvalue_gap_min, ...}
    alpha_star: float
    rho_A_at_alpha_star: float
    brent_diagnostics: dict


def sweep_rho_A(W_rec: np.ndarray, W_in: np.ndarray, W_out: np.ndarray,
                tau: float, n_points: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Compute ρ(A(α)) at n_points equally-spaced α ∈ [0, 1]."""
    alphas = np.linspace(0.0, 1.0, n_points)
    rhos = np.array([rho_A(W_rec, W_in, W_out, tau, float(a)) for a in alphas])
    return alphas, rhos


def check_monotonicity(alphas: np.ndarray, rhos: np.ndarray) -> dict:
    """
    Verify ρ(A(α)) is monotonically non-increasing across the sweep.
    Return {monotone: bool, worst_gap: float, gap_index: int}.
    A `worst_gap > 0` means the sweep went UP between consecutive points
    (violation of (V3)).
    """
    diffs = np.diff(rhos)  # rho_{k+1} - rho_k
    worst_gap = float(np.max(diffs))
    idx = int(np.argmax(diffs))
    return {
        "monotone_non_increasing": worst_gap <= 1e-10,
        "worst_gap": worst_gap,
        "worst_gap_alpha_from": float(alphas[idx]),
        "worst_gap_alpha_to": float(alphas[idx + 1]),
    }


def compute_alpha(W_rec: np.ndarray, W_in: np.ndarray, W_out: np.ndarray,
                  tau: float,
                  sweep_n_points: int = 21,
                  xtol: float = 1e-8) -> VerificationReport:
    """
    Full pipeline: verify hypotheses, sweep ρ(A(α)), verify assumption (E)
    diagnostically, solve for α* by bisection.

    Raises ValueError if (H1) or (H2) fail (theorem inapplicable).

    Returns a VerificationReport that carries every intermediate quantity a
    reviewer might want to inspect.
    """
    # (V1) H1: ρ(W_rec) < 1
    rho_W = spectral_radius(W_rec)
    H1 = rho_W < 1.0

    # (V2) H2: ρ(A(0)) > 1
    rho_A0 = rho_A(W_rec, W_in, W_out, tau, 0.0)
    H2 = rho_A0 > 1.0

    if not H1:
        raise ValueError(f"(H1) violated: ρ(W_rec) = {rho_W:.4f} ≥ 1. "
                         "Rescale W_rec (via target_sr) below 1 and retry.")
    if not H2:
        raise ValueError(f"(H2) violated: ρ(A(0)) = {rho_A0:.4f} ≤ 1. "
                         "Feedback is too weak to be supercritical at α=0. "
                         "Rescale W_in or W_out upward and retry.")

    # (V3) monotonicity sweep across [0, 1]
    alphas_grid, rhos_grid = sweep_rho_A(W_rec, W_in, W_out, tau, sweep_n_points)
    mono = check_monotonicity(alphas_grid, rhos_grid)

    # Assumption (E) sampled at each grid point
    E_diags = []
    for a in alphas_grid:
        E_diags.append(perturbation_derivative(W_rec, W_in, W_out, tau, float(a)))

    e_i_min_gap = float(min(d["eigenvalue_gap_A"] for d in E_diags))
    e_ii_min_inner = float(min(d["E_ii_inner_product_real"] for d in E_diags))
    e_i_holds = all(d["E_i_simple_dominant"] for d in E_diags)
    e_ii_holds = e_ii_min_inner > 0.0

    # Bisection for α*
    def g(a: float) -> float:
        return rho_A(W_rec, W_in, W_out, tau, a) - 1.0

    alpha_star, brent = _bisect(g, 0.0, 1.0, xtol=xtol)
    rho_at_star = rho_A(W_rec, W_in, W_out, tau, alpha_star)

    return VerificationReport(
        H1_rho_W_rec=float(rho_W),
        H1_holds=bool(H1),
        H2_rho_A_at_0=float(rho_A0),
        H2_holds=bool(H2),
        monotonicity_sweep={
            "alphas": alphas_grid.tolist(),
            "rho_A": rhos_grid.tolist(),
            **mono,
        },
        E_diagnostics={
            "per_alpha": E_diags,
            "E_i_min_eigenvalue_gap": e_i_min_gap,
            "E_i_holds": e_i_holds,
            "E_ii_min_inner_product_real": e_ii_min_inner,
            "E_ii_holds": e_ii_holds,
            "assumption_E_holds": e_i_holds and e_ii_holds,
        },
        alpha_star=float(alpha_star),
        rho_A_at_alpha_star=float(rho_at_star),
        brent_diagnostics=brent,
    )


def report_to_dict(report: VerificationReport) -> dict:
    """Serialise a VerificationReport to a plain dict for JSON storage."""
    return asdict(report)


# ── Joint scale + α search ────────────────────────────────────────────────────
#
# The theorem requires (H2): ρ(A(0)) > 1. This is a condition on the feedback
# pathway strength W_in · W_out relative to W_rec. If W_in and W_out are too
# small, ρ(A(0)) stays below 1 for all α — the linearised operator is
# subcritical everywhere and no α* exists. In that case, the theorem tells us
# nothing about α.
#
# The natural repair is to rescale W_in and W_out jointly by a factor s so
# that (H2) holds. Uniqueness (via monotonicity of ρ(A(α))) is a stronger
# condition and requires a somewhat larger s. This helper searches for the
# smallest s that makes both hold.


def _check_scale(W_rec: np.ndarray, W_in_unit: np.ndarray, W_out_unit: np.ndarray,
                 tau: float, s: float, n_points: int = 21) -> dict:
    """Evaluate the theorem's hypotheses at scaled weights (s·W_in_unit, s·W_out_unit)."""
    W_in = s * W_in_unit
    W_out = s * W_out_unit
    rho_A0 = rho_A(W_rec, W_in, W_out, tau, 0.0)
    alphas, rhos = sweep_rho_A(W_rec, W_in, W_out, tau, n_points)
    mono = check_monotonicity(alphas, rhos)
    return {
        "s": float(s),
        "rho_A_at_0": float(rho_A0),
        "H2_holds": bool(rho_A0 > 1.0),
        "monotone_non_increasing": bool(mono["monotone_non_increasing"]),
        "worst_gap": float(mono["worst_gap"]),
        "min_rho_A": float(np.min(rhos)),
        "max_rho_A": float(np.max(rhos)),
    }


def find_valid_scale(W_rec: np.ndarray, W_in_unit: np.ndarray, W_out_unit: np.ndarray,
                     tau: float,
                     s_lo: float = 0.05, s_hi: float = 2.0,
                     grid: int = 40, tol: float = 1e-3,
                     n_sweep: int = 41) -> dict:
    """
    Search for the smallest joint scale s ∈ [s_lo, s_hi] such that the
    scaled weights (s·W_in_unit, s·W_out_unit) satisfy:
        (H2)                            ρ(A(0)) > 1
        empirical monotonicity of α → ρ(A(α))  (via sweep at `n_sweep` points)

    Strategy: log-spaced grid over [s_lo, s_hi]. Find the smallest s where
    both hold. Then bisect between the previous grid point and that s to
    localise the transition to precision `tol`.

    Returns a dict with:
        s_star:         smallest valid scale found
        s_grid:         list of grid s values tested
        grid_results:   per-s diagnostic dicts
        transition_H2:  smallest s where (H2) alone holds
        transition_mono: smallest s where monotonicity alone holds
        transition_both: smallest s where both hold (= s_star)
    """
    s_grid = np.geomspace(s_lo, s_hi, grid)
    results = [_check_scale(W_rec, W_in_unit, W_out_unit, tau, float(s), n_sweep)
               for s in s_grid]

    def first_true(pred):
        for i, r in enumerate(results):
            if pred(r):
                return i
        return None

    idx_h2 = first_true(lambda r: r["H2_holds"])
    idx_mono = first_true(lambda r: r["monotone_non_increasing"])
    idx_both = first_true(lambda r: r["H2_holds"] and r["monotone_non_increasing"])

    def bisect_bracket(idx_hit: int | None) -> float | None:
        if idx_hit is None:
            return None
        if idx_hit == 0:
            return float(s_grid[0])
        lo = float(s_grid[idx_hit - 1])
        hi = float(s_grid[idx_hit])
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            r = _check_scale(W_rec, W_in_unit, W_out_unit, tau, mid, n_sweep)
            hit = r["H2_holds"] and r["monotone_non_increasing"]
            if hit:
                hi = mid
            else:
                lo = mid
        return hi

    s_star = bisect_bracket(idx_both)
    return {
        "s_star": s_star,
        "s_grid": s_grid.tolist(),
        "grid_results": results,
        "transition_H2_s": (float(s_grid[idx_h2]) if idx_h2 is not None else None),
        "transition_mono_s": (float(s_grid[idx_mono]) if idx_mono is not None else None),
        "transition_both_s": s_star,
    }
