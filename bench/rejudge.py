from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import time

from .dataset import iter_tests
from .personas import build_system_prompt
from .judge import render_judge_prompt, call_judge
from .storage import RunStore

ISO = "%Y-%m-%dT%H%M%S.%fZ"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def _fmt_dur(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:d}m{s:02d}s"


class RateLimiter:
    def __init__(self, rps: float) -> None:
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            import time as _t

            now = _t.monotonic()
            delta = now - self._last
            wait_s = self.min_interval - delta
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._last = _t.monotonic()


@dataclass
class RejudgeConfig:
    run_id: str
    dataset_path: str
    output_dir: str
    judge_model: str
    concurrency: int
    rate_limit_rps: float
    retries: int
    models_filter: Optional[List[str]] = None
    personas_filter: Optional[List[str]] = None


async def rejudge(run_dir: Path, config: RejudgeConfig, logger) -> None:
    store = RunStore(run_dir)
    store.ensure()

    # Load dataset index by id
    id_to_test: Dict[str, Dict[str, Any]] = {}
    for t in iter_tests(config.dataset_path):
        tid = str(t.get("id"))
        id_to_test[tid] = t

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found in {run_dir}")
    df = pd.read_csv(results_csv)

    # Select candidates with missing/errored judge
    def is_missing_judge(row: pd.Series) -> bool:
        j = row.get("judge_json")
        err = row.get("error")
        has_candidate = isinstance(row.get("raw_candidate"), str) and len(row.get("raw_candidate")) > 0
        missing = pd.isna(j) or j == "" or (isinstance(j, float) and pd.isna(j))
        return has_candidate and (missing or (isinstance(err, str) and len(err) > 0))

    mask = df.apply(is_missing_judge, axis=1)
    if config.models_filter:
        mask = mask & df["model_slug"].isin(config.models_filter)
    if config.personas_filter:
        mask = mask & df["persona_slug"].isin(config.personas_filter)

    todo = df[mask].copy()
    logger.info("rejudge: selected %s rows", len(todo))
    if todo.empty:
        return

    limiter = RateLimiter(config.rate_limit_rps)
    sem = asyncio.Semaphore(config.concurrency)

    # Progress tracking
    start_mono = time.monotonic()
    total_tasks = len(todo)
    done_tasks = 0
    progress_lock = asyncio.Lock()

    async def log_progress() -> None:
        nonlocal done_tasks, total_tasks
        elapsed = time.monotonic() - start_mono
        pct = (done_tasks / total_tasks * 100.0) if total_tasks else 0.0
        remaining = max(0, total_tasks - done_tasks)
        eta_s = (elapsed / done_tasks * remaining) if done_tasks else 0.0
        logger.info(
            "rejudge progress %d/%d (%.1f%%) elapsed=%s eta=%s",
            done_tasks,
            total_tasks,
            pct,
            _fmt_dur(elapsed),
            _fmt_dur(eta_s),
        )

    async def worker(idx: int, row: pd.Series) -> Dict[str, Any]:
        nonlocal done_tasks, total_tasks
        test_id = str(row.get("test_id"))
        persona_slug = str(row.get("persona_slug"))
        model_slug = str(row.get("model_slug"))
        category = str(row.get("category"))
        subcategory = str(row.get("subcategory"))
        candidate = str(row.get("raw_candidate") or "")
        started_at = _iso_now()
        error: Optional[str] = None
        judge_json: Optional[Dict[str, Any]] = None
        attack_success: Optional[bool] = None
        overall: Optional[float] = None
        safety_score: Optional[float] = None

        try:
            test = id_to_test.get(test_id)
            if not test:
                raise KeyError(f"test_id not found in dataset: {test_id}")
            # Rebuild conversation
            conversation = list(test["conversation"])  # shallow copy
            system_content = build_system_prompt(str(test["persona_system_prompt"]), persona_slug)
            if conversation and conversation[0].get("role") == "system":
                conversation[0] = {"role": "system", "content": system_content}
            else:
                conversation = [{"role": "system", "content": system_content}] + conversation
            user_prompt = next(
                (m["content"] for m in reversed(conversation) if m.get("role") == "user"),
                "",
            )
            judge_prompt = render_judge_prompt(str(test["judge_prompt"]), conversation, user_prompt, candidate)

            async with sem:
                await limiter.acquire()
                judge_json = await call_judge(config.judge_model, judge_prompt, api_log_path=str(run_dir / "api_logs" / f"judge_{config.judge_model}.log"))

            if isinstance(judge_json, dict):
                attack_success = judge_json.get("attack_success")
                overall = judge_json.get("overall")
                safety = judge_json.get("safety") or {}
                if isinstance(safety, dict):
                    val = safety.get("score")
                    if isinstance(val, (int, float)):
                        safety_score = float(val)

        except Exception as exc:
            error = str(exc)

        completed_at = _iso_now()
        out = {
            "run_id": config.run_id,
            "test_id": test_id,
            "category": category,
            "subcategory": subcategory,
            "persona_slug": persona_slug,
            "model_slug": model_slug,
            "started_at": started_at,
            "completed_at": completed_at,
            "raw_candidate": candidate,
            "judge_json": judge_json,
            "attack_success": attack_success,
            "overall": overall,
            "safety_score": safety_score,
            "error": error,
            "rejudge": True,
        }
        store.append_partial(out)
        logger.info("rejudged id=%s persona=%s model=%s overall=%s", test_id, persona_slug, model_slug, overall)
        async with progress_lock:
            done_tasks += 1
            await log_progress()
        return out

    tasks: List[asyncio.Task[Dict[str, Any]]] = []
    for i, row in todo.iterrows():
        tasks.append(asyncio.create_task(worker(i, row)))
    logger.info("rejudge scheduled %d tasks", total_tasks)

    new_rows: List[Dict[str, Any]] = []
    if tasks:
        for r in await asyncio.gather(*tasks):
            new_rows.append(r)

    # Merge back into results.csv and results.parquet
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        key_cols = ["test_id", "persona_slug", "model_slug"]
        df_keyed = df.set_index(key_cols)
        df_new_keyed = df_new.set_index(key_cols)
        # Update fields
        for col in ["judge_json", "attack_success", "overall", "safety_score", "error"]:
            if col in df_new_keyed.columns:
                df_keyed.loc[df_new_keyed.index, col] = df_new_keyed[col]
        merged = df_keyed.reset_index()
        store.write_final(merged.to_dict(orient="records"))
