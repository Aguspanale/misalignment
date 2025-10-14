from __future__ import annotations

import csv
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class RunStore:
    run_dir: Path

    @property
    def partials_path(self) -> Path:
        return self.run_dir / "partials.jsonl"

    @property
    def results_csv_path(self) -> Path:
        return self.run_dir / "results.csv"

    @property
    def results_parquet_path(self) -> Path:
        return self.run_dir / "results.parquet"

    @property
    def state_sqlite_path(self) -> Path:
        return self.run_dir / "state.sqlite"

    @property
    def state_json_path(self) -> Path:
        return self.run_dir / "state.json"

    def ensure(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def append_partial(self, record: Dict) -> None:
        self.ensure()
        with open(self.partials_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_final(self, rows: List[Dict]) -> None:
        self.ensure()
        if not rows:
            # Still create empty files
            pd.DataFrame([]).to_csv(self.results_csv_path, index=False)
            pd.DataFrame([]).to_parquet(self.results_parquet_path, index=False)
            return
        df = pd.json_normalize(rows, sep=".")
        df.to_csv(self.results_csv_path, index=False)
        df.to_parquet(self.results_parquet_path, index=False)

    # ---- Resume state ----
    def mark_done(self, key: Tuple[str, str, str]) -> None:
        key_str = "|".join(key)
        if self._sqlite_available():
            self._sqlite_mark_done(key_str)
        else:
            self._json_mark_done(key_str)

    def is_done(self, key: Tuple[str, str, str]) -> bool:
        key_str = "|".join(key)
        if self._sqlite_available():
            return self._sqlite_is_done(key_str)
        return self._json_is_done(key_str)

    def _sqlite_available(self) -> bool:
        # Prefer sqlite; create if missing
        return True

    @contextmanager
    def _sqlite_conn(self):
        self.ensure()
        conn = sqlite3.connect(self.state_sqlite_path)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS done (key TEXT PRIMARY KEY)")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _sqlite_mark_done(self, key_str: str) -> None:
        with self._sqlite_conn() as conn:
            conn.execute("INSERT OR IGNORE INTO done(key) VALUES (?)", (key_str,))

    def _sqlite_is_done(self, key_str: str) -> bool:
        with self._sqlite_conn() as conn:
            cur = conn.execute("SELECT 1 FROM done WHERE key=?", (key_str,))
            return cur.fetchone() is not None

    # JSON fallback
    def _json_mark_done(self, key_str: str) -> None:
        self.ensure()
        state = set()
        if self.state_json_path.exists():
            with open(self.state_json_path, "r", encoding="utf-8") as f:
                state = set(json.load(f))
        state.add(key_str)
        with open(self.state_json_path, "w", encoding="utf-8") as f:
            json.dump(sorted(state), f, ensure_ascii=False)

    def _json_is_done(self, key_str: str) -> bool:
        if not self.state_json_path.exists():
            return False
        with open(self.state_json_path, "r", encoding="utf-8") as f:
            state = set(json.load(f))
        return key_str in state
