from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .storage import RunStore


@dataclass
class SalvageConfig:
    run_id: str
    output_dir: str


def salvage_from_partials(run_dir: Path) -> List[Dict[str, Any]]:
    store = RunStore(run_dir)
    partials_path = store.partials_path
    if not partials_path.exists():
        raise FileNotFoundError(f"partials.jsonl not found at {partials_path}")

    latest: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    with open(partials_path, "r", encoding="utf-8") as f:
        for line in f:
            print(line)
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = (
                str(obj.get("test_id")),
                str(obj.get("persona_slug")),
                str(obj.get("model_slug")),
            )
            latest[key] = obj

    rows = list(latest.values())
    return rows


def write_finals_from_partials(run_dir: Path) -> None:
    rows = salvage_from_partials(run_dir)
    store = RunStore(run_dir)
    store.write_final(rows)
