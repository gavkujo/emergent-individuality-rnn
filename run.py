"""
Single entry point for all experiments.

Usage:
    python run.py --list              # list available experiments
    python run.py <name>              # run experiment <name>

Experiments live in src/experiments/<name>.py and must expose:
    NAME, DESCRIPTION, DETAILS, VERSION, run(device) -> dict|None

If run() returns a dict, it is saved via src.results_io.save_result with the
module's NAME/DESCRIPTION/DETAILS/VERSION. If run() returns None, nothing is
saved (used for figure-generation and other side-effect experiments).
"""

import argparse
import importlib
import os
import pkgutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.results_io import save_result


def discover_experiments():
    """Return a dict {name: module_path} for everything in src/experiments/."""
    pkg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "src", "experiments")
    found = {}
    for _, modname, _ in pkgutil.iter_modules([pkg_path]):
        if modname.startswith("_"):
            continue
        found[modname] = f"src.experiments.{modname}"
    return found


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    experiments = discover_experiments()

    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument("name", nargs="?",
                        help="Experiment name (one of: " +
                             ", ".join(sorted(experiments)) + ")")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments and exit.")
    parser.add_argument("--device", default=None,
                        help="Override device (default: auto-detect).")
    args = parser.parse_args()

    if args.list or not args.name:
        if not experiments:
            print("(no experiments found in src/experiments/)")
            return
        print("Available experiments:\n")
        for name in sorted(experiments):
            mod = importlib.import_module(experiments[name])
            desc = getattr(mod, "DESCRIPTION", "(no description)")
            print(f"  {name:20s} {desc}")
        return

    if args.name not in experiments:
        print(f"Unknown experiment: {args.name}")
        print(f"Available: {', '.join(sorted(experiments))}")
        sys.exit(1)

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    mod = importlib.import_module(experiments[args.name])
    name = getattr(mod, "NAME", args.name)
    description = getattr(mod, "DESCRIPTION", "")
    details = getattr(mod, "DETAILS", "")
    version = getattr(mod, "VERSION", "v1")

    print(f"\n=== {name} ({version}) ===")
    print(f"{description}\n")

    t0 = time.time()
    results = mod.run(device)
    elapsed = time.time() - t0
    print(f"\n(elapsed: {elapsed:.1f}s)")

    if results is None:
        print("(no results dict returned, nothing saved)")
        return

    path = save_result(
        script_file=__file__,
        name=name,
        description=description,
        details=details,
        results=results,
        version=version,
    )
    print(f"✓ Saved to {path}")


if __name__ == "__main__":
    main()
