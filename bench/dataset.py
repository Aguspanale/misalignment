from __future__ import annotations

from typing import Dict, Iterator, List

import json

REQUIRED_KEYS = {
    "id",
    "category",
    "subcategory",
    "persona_system_prompt",
    "conversation",
    "expected_safe_behavior",
    "attack_notes",
    "judge_prompt",
}


def _validate_item(item: Dict) -> None:
    missing = REQUIRED_KEYS - set(item.keys())
    if missing:
        raise ValueError(f"Dataset item missing keys: {sorted(missing)}")
    if not isinstance(item.get("conversation"), list):
        raise ValueError("conversation must be a list of message dicts")


def iter_tests(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            _validate_item(item)
            yield item
