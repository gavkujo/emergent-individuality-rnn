"""
Result-saving and -loading helpers.

Every experiment writes a JSON via save_result. Results are organised by
ISO week:

    results/
      week<NN>_<YYYY>/
        <name>_<YYYYMMDD>_<version>.json

Schema:
    name        : experiment name (matches filename prefix)
    description : what the experiment does (one-liner)
    details     : what changed from previous versions
    time        : ISO-8601 timestamp, second precision
    results     : actual experiment data (JSON-serialisable)

load_latest searches across ALL week folders and returns the most recently
dated file matching a given name.
"""

import os
import json
import glob
from datetime import datetime
from typing import Optional


def _week_folder(dt: datetime) -> str:
    """ISO week folder name: weekNN_YYYY (e.g. week22_2026)."""
    iso_year, iso_week, _ = dt.isocalendar()
    return f"week{iso_week:02d}_{iso_year}"


def save_result(script_file: str, name: str, description: str,
                details: str, results: dict, version: str = "v1") -> str:
    """
    Write results JSON to <repo_root>/results/<week_folder>/<name>_<date>_<version>.json.

    Args:
        script_file: caller's __file__. Used to find repo root.
        name:        experiment name (filename prefix)
        description: short description of what the experiment does
        details:     what changed from previous versions of this experiment
        results:     experiment data (any JSON-serialisable structure)
        version:     version tag (default 'v1')

    Returns:
        absolute path to the saved JSON
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    timestamp = now.isoformat(timespec="seconds")
    week_dir = _week_folder(now)

    repo_root = os.path.dirname(os.path.abspath(script_file))
    out_dir = os.path.join(repo_root, "results", week_dir)
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{name}_{date_str}_{version}.json"
    path = os.path.join(out_dir, filename)

    payload = {
        "name": name,
        "description": description,
        "details": details,
        "time": timestamp,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_latest(name: str, repo_root: Optional[str] = None) -> dict:
    """
    Load the most recently dated results JSON matching <name>_*.json,
    searching across all week folders.

    Returns the full payload (with name, description, details, time, results).
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(repo_root, "results", "**", f"{name}_*.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"no results found matching {pattern}")
    # sort by the date embedded in the filename (after the name prefix)
    # filename: <name>_<YYYYMMDD>_<version>.json
    files.sort(key=lambda p: os.path.basename(p))
    with open(files[-1]) as f:
        return json.load(f)
