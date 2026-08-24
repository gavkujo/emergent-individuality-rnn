"""
Seeded experiment harness with per-seed checkpointing.

Purpose: every hardened experiment runs N seeds independently, saves each
seed's output as a JSON so a killed run doesn't lose completed work, then
aggregates. This module holds that plumbing so experiment code stays
focused on the science.

Usage:

    from src.harness import run_seeded

    def per_seed(seed, device, config):
        # returns a JSON-serialisable dict, one entry per seed
        ...

    def aggregate(per_seed_results, config):
        # per_seed_results: {seed: dict}
        # returns the final results dict for save_result
        ...

    final, path = run_seeded(
        name="decoder_hardened",
        version="v1",
        seeds=list(range(30)),
        per_seed_fn=per_seed,
        aggregate_fn=aggregate,
        config=CONFIG,
        device=device,
        description=DESCRIPTION,
        details=DETAILS,
    )
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime
from typing import Callable, Optional

import torch

from src.results_io import save_result


# `harness.py` lives at `<repo>/src/harness.py`, so the repo root is two
# levels up. Every experiment writes into `<repo>/results/`, so the
# harness resolves the checkpoint / save location relative to itself
# rather than relative to whichever module happens to call it.
_REPO_ROOT_MARKER = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "run.py"))


# ── Environment capture ──────────────────────────────────────────────────────

def _capture_environment(device) -> dict:
    """Reproducibility metadata written into every checkpoint + final JSON."""
    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "device": str(device),
    }
    dev_str = str(device)
    if dev_str.startswith("cuda") and torch.cuda.is_available():
        env["cuda_device_name"] = torch.cuda.get_device_name(0)
        env["cuda_version"] = torch.version.cuda
    elif dev_str.startswith("mps"):
        env["backend"] = "MPS (Apple Silicon)"
    return env


# ── Checkpoint I/O ───────────────────────────────────────────────────────────

def _checkpoint_dir(name: str, version: str) -> str:
    """`<repo_root>/results/_checkpoints/<name>_<version>/`."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "results", "_checkpoints",
                        f"{name}_{version}")


def _seed_ckpt_path(ckpt_dir: str, seed: int) -> str:
    return os.path.join(ckpt_dir, f"seed_{seed:04d}.json")


def _load_seed_checkpoint(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_seed_checkpoint(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp, path)  # atomic on POSIX


def _json_default(o):
    """JSON fallback for numpy scalars / arrays."""
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
    except ImportError:
        pass
    if isinstance(o, torch.device):
        return str(o)
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


# ── Seeded runner ────────────────────────────────────────────────────────────

def run_seeded(name: str,
               version: str,
               seeds: list[int],
               per_seed_fn: Callable[[int, "torch.device", dict], dict],
               aggregate_fn: Callable[[dict[int, dict], dict], dict],
               config: dict,
               device,
               description: str = "",
               details: str = "",
               verbose: bool = True) -> tuple[dict, str]:
    """
    Run `per_seed_fn(seed, device, config)` for every seed, checkpointing
    each one to disk, then aggregate and save the final JSON via
    `save_result`.

    Args:
        name, version:   passed to save_result; also identify the
                         checkpoint dir.
        seeds:           list of int seeds to run.
        per_seed_fn:     called for each seed. Must return a
                         JSON-serialisable dict.
        aggregate_fn:    called once at the end with
                         `{seed: per_seed_result}` and the config.
                         Must return the final results dict.
        config:          experiment-level config dict (JSON-serialisable).
                         Embedded in checkpoint + final JSON.
        device:          torch device.
        description:     save_result field.
        details:         save_result field.
        verbose:         log per-seed progress.

    Returns:
        (final_results_dict, saved_path)
    """
    if not seeds:
        raise ValueError("run_seeded called with empty seed list")

    ckpt_dir = _checkpoint_dir(name, version)
    os.makedirs(ckpt_dir, exist_ok=True)

    env = _capture_environment(device)
    t_run_start = time.time()

    per_seed_results: dict[int, dict] = {}
    per_seed_wallclock: dict[int, float] = {}
    per_seed_status: dict[int, str] = {}

    n_seeds = len(seeds)
    if verbose:
        print(f"[harness] {name} {version}: {n_seeds} seeds, "
              f"device={device}, checkpoints @ {ckpt_dir}")

    for i, seed in enumerate(seeds, start=1):
        ck_path = _seed_ckpt_path(ckpt_dir, seed)
        cached = _load_seed_checkpoint(ck_path)
        if cached is not None:
            per_seed_results[seed] = cached["result"]
            per_seed_wallclock[seed] = float(cached.get("wallclock_s", 0.0))
            per_seed_status[seed] = "cached"
            if verbose:
                print(f"[harness]   ({i}/{n_seeds}) seed {seed}: "
                      f"cached ({per_seed_wallclock[seed]:.1f}s)")
            continue

        if verbose:
            print(f"[harness]   ({i}/{n_seeds}) seed {seed}: running...",
                  flush=True)
        t0 = time.time()
        result = per_seed_fn(seed, device, config)
        wall = time.time() - t0

        payload = {
            "name":         name,
            "version":      version,
            "seed":         seed,
            "config":       config,
            "environment":  env,
            "time":         datetime.now().isoformat(timespec="seconds"),
            "wallclock_s":  wall,
            "result":       result,
        }
        _save_seed_checkpoint(ck_path, payload)

        per_seed_results[seed] = result
        per_seed_wallclock[seed] = wall
        per_seed_status[seed] = "computed"

        # Free device memory across seeds — the sleep_order_multiseed
        # slowdown was traced to accumulated device memory.
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        if verbose:
            print(f"[harness]   ({i}/{n_seeds}) seed {seed}: "
                  f"done in {wall:.1f}s", flush=True)

    t_run_end = time.time()

    # Aggregate
    if verbose:
        print(f"\n[harness] aggregating {len(per_seed_results)} seeds...")
    aggregated = aggregate_fn(per_seed_results, config)

    final_results = {
        "config":         config,
        "environment":    env,
        "seeds":          list(seeds),
        "n_seeds":        n_seeds,
        "wallclock_total_s":  float(t_run_end - t_run_start),
        "wallclock_per_seed_s": {int(k): float(v)
                                 for k, v in per_seed_wallclock.items()},
        "seed_status":    per_seed_status,
        "aggregated":     aggregated,
        "per_seed":       per_seed_results,
    }

    # Point save_result at `<repo>/run.py` so it resolves the repo root
    # correctly regardless of which experiment module called us.
    saved_path = save_result(
        script_file=_REPO_ROOT_MARKER,
        name=name,
        description=description,
        details=details,
        results=final_results,
        version=version,
    )
    if verbose:
        print(f"[harness] saved → {saved_path}")
    return final_results, saved_path
