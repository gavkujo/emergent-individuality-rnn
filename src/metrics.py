"""
Metrics and statistical helpers used across experiments.

Divergence metrics:
    subspace_angle(sa, sb, n)     — angle between top-n PCs
    effective_dim(states, tau)    — smallest n reaching variance threshold
    l2_divergence(sa, sb)         — mean of ‖s_a(t) - s_b(t)‖ over time
    trajectory_distance(sa, sb)   — sinkhorn-free "shape" distance

Decoder:
    decoder_accuracy(states_by_label, ...) — held-out k-fold decoder,
        with optional ridge or MLP head. Returns dict with mean, std,
        in-sample, and configuration.

Statistical:
    bootstrap_ci_mean(vals, ...)  — CI of the mean via percentile bootstrap
    bootstrap_ci_slope(x, y_seeds, ...) — CI of an OLS slope over
        per-seed curves (each row = one seed's y vs x).
    paired_delta_ci(a, b, ...)    — CI of the paired mean difference a - b
    sign_test(a, b)               — count of seeds where a > b

Every function is pure numpy. Nothing here uses torch or a device.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


# ── Divergence metrics ──────────────────────────────────────────────────────

def subspace_angle(sa: np.ndarray, sb: np.ndarray, n: int = 2) -> float:
    """Mean principal angle between the top-n PCs of two trajectories, in degrees.

    Args:
        sa, sb: (T, hidden_dim) idle trajectories.
        n:      subspace dimension. Default 2 — this is the effective
                dimension of the trained idle limit cycle at sinusoid
                tasks (see subspace_dim finding). Under other task
                families, verify with `effective_dim` first.
    """
    a = sa - sa.mean(0)
    b = sb - sb.mean(0)
    _, _, Va = np.linalg.svd(a, full_matrices=False)
    _, _, Vb = np.linalg.svd(b, full_matrices=False)
    sv = np.clip(np.linalg.svd(Va[:n] @ Vb[:n].T, compute_uv=False), -1.0, 1.0)
    return float(np.arccos(sv).mean() * 180.0 / np.pi)


def effective_dim(states: np.ndarray, variance_threshold: float = 0.99) -> int:
    """Smallest n such that cum(sv²)/total ≥ variance_threshold."""
    centered = states - states.mean(0)
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    var = sv ** 2
    cum = np.cumsum(var) / max(float(var.sum()), 1e-30)
    return int(min(np.searchsorted(cum, variance_threshold) + 1, len(sv)))


def l2_divergence(sa: np.ndarray, sb: np.ndarray) -> float:
    """Mean over time of ‖s_a(t) - s_b(t)‖. Requires equal-length inputs."""
    return float(np.linalg.norm(sa - sb, axis=-1).mean())


# ── Decoder ─────────────────────────────────────────────────────────────────

def _ridge_fit_predict(Xtr, Ytr, Xte, ridge_lambda: float) -> np.ndarray:
    """Closed-form ridge regression on one-hot targets; return argmax predictions."""
    d = Xtr.shape[1]
    XtX = Xtr.T @ Xtr + ridge_lambda * np.eye(d)
    W = np.linalg.solve(XtX, Xtr.T @ Ytr)
    return (Xte @ W).argmax(1)


def _mlp_fit_predict(Xtr, ytr, Xte, hidden: int = 64,
                     lr: float = 1e-2, epochs: int = 200,
                     seed: int = 0) -> np.ndarray:
    """Small MLP classifier fit with SGD; return argmax predictions on Xte.

    Kept dependency-light: pure numpy with a single hidden layer. Not a
    performance-optimised implementation — meant for auxiliary validation
    that the ridge accuracy isn't linearly-decoder-limited.
    """
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape
    K = int(ytr.max() + 1)
    W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, K)) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(K)

    Y = np.eye(K)[ytr]
    for _ in range(epochs):
        # forward
        z1 = Xtr @ W1 + b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ W2 + b2
        # softmax
        z2 = z2 - z2.max(axis=1, keepdims=True)
        p = np.exp(z2)
        p /= p.sum(axis=1, keepdims=True)
        # gradient (cross-entropy on one-hot)
        dz2 = (p - Y) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = Xtr.T @ dz1
        db1 = dz1.sum(0)
        # step
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    # predict
    z1 = Xte @ W1 + b1
    a1 = np.maximum(z1, 0)
    z2 = a1 @ W2 + b2
    return z2.argmax(1)


def decoder_accuracy(states_by_label: dict, n_splits: int = 5,
                     ridge_lambda: float = 1.0,
                     trim: int = 150,
                     head: str = "ridge",
                     shuffle_time: bool = False,
                     seed: int = 0) -> dict:
    """Held-out k-fold decoder accuracy on per-label trajectories.

    Args:
        states_by_label: {label: ndarray of shape (T, hidden_dim)}. Labels
            can be any hashable; the function sorts them and assigns
            0..K-1 for the classifier.
        n_splits:    k for k-fold cross-validation.
        ridge_lambda: L2 regularisation for the ridge head.
        trim:        take the last `trim` timesteps of each trajectory
            as the class samples (transient suppression).
        head:        "ridge" or "mlp". Ridge is closed-form and fast;
            MLP catches non-linear decodability.
        shuffle_time: if True, shuffle the trimmed timestep index inside
            each class before pooling. Used for the shuffled-trajectory
            CONTROL: strips temporal structure while keeping marginals.
        seed:        RNG seed for fold shuffling and MLP init.

    Returns:
        dict with:
            mean        mean held-out accuracy across folds
            std         std of held-out accuracy across folds
            in_sample   train-on-all / test-on-all fit (comparison
                        with the old bug)
            n_splits, n_classes, n_per_class, ridge_lambda, head,
            shuffle_time
    """
    if head not in {"ridge", "mlp"}:
        raise ValueError(f"unknown decoder head: {head!r}")

    labels = sorted(states_by_label.keys())
    n_classes = len(labels)
    Y_oh = np.eye(n_classes)

    segments = [states_by_label[l][-trim:] for l in labels]
    n_per_class = segments[0].shape[0]
    if any(s.shape[0] != n_per_class for s in segments):
        raise ValueError("inconsistent trimmed length across labels")

    rng = np.random.default_rng(seed)
    if shuffle_time:
        segments = [s[rng.permutation(n_per_class)] for s in segments]

    X = np.vstack(segments)                                       # (K*trim, d)
    y = np.concatenate([[i] * n_per_class for i in range(n_classes)])
    Y = Y_oh[y]
    n, d = X.shape

    # In-sample fit for reporting the old metric
    if head == "ridge":
        preds_in = _ridge_fit_predict(X, Y, X, ridge_lambda)
    else:
        preds_in = _mlp_fit_predict(X, y, X, seed=seed)
    in_sample = float((preds_in == y).mean())

    # Stratified across the trimmed index
    fold_assignments = np.empty(n, dtype=int)
    for c in range(n_classes):
        idx = np.arange(c * n_per_class, (c + 1) * n_per_class)
        rng.shuffle(idx)
        for k, sl in enumerate(np.array_split(idx, n_splits)):
            fold_assignments[sl] = k

    fold_accs = []
    for k in range(n_splits):
        test_mask = fold_assignments == k
        train_mask = ~test_mask
        if head == "ridge":
            preds = _ridge_fit_predict(X[train_mask], Y[train_mask],
                                       X[test_mask], ridge_lambda)
        else:
            preds = _mlp_fit_predict(X[train_mask], y[train_mask],
                                     X[test_mask], seed=seed + k)
        fold_accs.append(float((preds == y[test_mask]).mean()))

    fold_accs = np.array(fold_accs)
    return {
        "mean":          float(fold_accs.mean()),
        "std":           float(fold_accs.std(ddof=1)) if n_splits > 1 else 0.0,
        "in_sample":     in_sample,
        "n_splits":      n_splits,
        "n_classes":     n_classes,
        "n_per_class":   n_per_class,
        "ridge_lambda":  ridge_lambda,
        "head":          head,
        "shuffle_time":  shuffle_time,
    }


# ── Statistical helpers ─────────────────────────────────────────────────────

def bootstrap_ci_mean(vals: np.ndarray, n_resamples: int = 2000,
                      ci_level: float = 0.95,
                      seed: int = 0) -> dict:
    """Percentile-bootstrap CI on the mean of a 1-D array."""
    vals = np.asarray(vals)
    rng = np.random.default_rng(seed)
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "ci_level": ci_level,
                "n_resamples": n_resamples, "n": 0}
    means = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[r] = vals[idx].mean()
    lo = 0.5 - ci_level / 2.0
    hi = 0.5 + ci_level / 2.0
    return {
        "mean":         float(vals.mean()),
        "std":          float(vals.std(ddof=1)) if n > 1 else 0.0,
        "sem":          float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "ci_lo":        float(np.quantile(means, lo)),
        "ci_hi":        float(np.quantile(means, hi)),
        "ci_level":     ci_level,
        "n_resamples":  n_resamples,
        "n":            int(n),
    }


def bootstrap_ci_slope(x: np.ndarray, y_by_seed: np.ndarray,
                       n_resamples: int = 2000, ci_level: float = 0.95,
                       seed: int = 0) -> dict:
    """CI on the OLS slope of y vs x, bootstrapping across seeds.

    Args:
        x:         (n_points,) x-values (e.g. cycle indices).
        y_by_seed: (n_seeds, n_points) — one row per seed.
        n_resamples, ci_level, seed: bootstrap knobs.

    Slope is fit per resample by averaging y across the resampled seeds
    and running polyfit degree 1.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_by_seed, dtype=float)
    n_seeds = y.shape[0]
    rng = np.random.default_rng(seed)

    def _slope(vals):
        return float(np.polyfit(x, vals, 1)[0])

    point = _slope(y.mean(0))
    slopes = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        slopes[r] = _slope(y[idx].mean(0))
    lo = 0.5 - ci_level / 2.0
    hi = 0.5 + ci_level / 2.0
    return {
        "slope_point":  point,
        "ci_lo":        float(np.quantile(slopes, lo)),
        "ci_hi":        float(np.quantile(slopes, hi)),
        "ci_level":     ci_level,
        "n_resamples":  n_resamples,
        "n_seeds":      int(n_seeds),
    }


def paired_delta_ci(a: np.ndarray, b: np.ndarray,
                    n_resamples: int = 2000, ci_level: float = 0.95,
                    seed: int = 0) -> dict:
    """CI on the paired mean difference (a - b) via percentile bootstrap."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    delta = a - b
    stats = bootstrap_ci_mean(delta, n_resamples, ci_level, seed=seed)
    stats.update({
        "a_mean": float(a.mean()),
        "b_mean": float(b.mean()),
        "n_seeds_a_greater": int(np.sum(a > b)),
        "n_seeds_b_greater": int(np.sum(b > a)),
    })
    return stats


def sign_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Non-parametric sign counts. Two-sided binomial p-value."""
    from math import comb
    a = np.asarray(a)
    b = np.asarray(b)
    n_pos = int(np.sum(a > b))
    n_neg = int(np.sum(a < b))
    n = n_pos + n_neg
    if n == 0:
        return {"n_positive": 0, "n_negative": 0, "n_ties": len(a),
                "p_two_sided": float("nan")}
    k = min(n_pos, n_neg)
    # two-sided binomial at p = 0.5
    tail = sum(comb(n, j) for j in range(k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_ties":     int(len(a) - n),
        "p_two_sided": float(p),
    }


def holm_bonferroni(pvals: dict) -> dict:
    """Holm-Bonferroni step-down correction on a dict of {name: p_value}.

    Returns a parallel dict of adjusted p-values. Multiple-comparison
    correction for reporting families of related tests.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted
