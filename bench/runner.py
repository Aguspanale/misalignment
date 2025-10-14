from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from rich.logging import RichHandler
from rich.console import Console
import logging
import sys
from pathlib import Path

from .config import RunConfig
from .dataset import iter_tests
from .judge import call_judge, render_judge_prompt
from .personas import build_system_prompt, get_all_personas
from .storage import RunStore
from .genai import chat as genai_chat


ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


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
            now = time.monotonic()
            delta = now - self._last
            wait_s = self.min_interval - delta
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._last = time.monotonic()


async def call_model(model_slug: str, messages: List[Dict[str, str]], *, api_log_path: Optional[str] = None) -> str:
    # Uses GenAI endpoint and env vars like dataset_builder.py, run in thread to avoid blocking loop
    return await asyncio.to_thread(
        genai_chat,
        model_slug,
        messages,
        response_format=None,
        temperature=0.2,
        max_tokens=1800,
        max_retries=3,
        api_log_path=api_log_path,
    )


@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=8), stop=stop_after_attempt(3))
async def call_model_with_retry(model_slug: str, messages: List[Dict[str, str]], *, api_log_path: Optional[str]) -> str:
    return await call_model(model_slug, messages, api_log_path=api_log_path)


@dataclass
class BenchmarkRunner:
    store: RunStore
    logger: logging.Logger

    async def run(self, config: RunConfig) -> List[Dict[str, Any]]:
        self.logger.info("starting run_id=%s dataset=%s models=%s", config.run_id, config.dataset_path, ",".join(config.models_under_test))
        limiter_model = RateLimiter(config.rate_limit_rps)
        limiter_judge = RateLimiter(config.rate_limit_rps)
        sem = asyncio.Semaphore(config.concurrency)

        results: List[Dict[str, Any]] = []

        api_logs_dir = self.store.run_dir / "api_logs"
        api_logs_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        start_mono = time.monotonic()
        progress_lock = asyncio.Lock()
        total_tasks = 0
        done_tasks = 0

        async def log_progress() -> None:
            nonlocal done_tasks, total_tasks
            elapsed = time.monotonic() - start_mono
            pct = (done_tasks / total_tasks * 100.0) if total_tasks else 0.0
            remaining = max(0, total_tasks - done_tasks)
            eta_s = (elapsed / done_tasks * remaining) if done_tasks else 0.0
            self.logger.info(
                "progress %d/%d (%.1f%%) elapsed=%s eta=%s",
                done_tasks,
                total_tasks,
                pct,
                _fmt_dur(elapsed),
                _fmt_dur(eta_s),
            )

        async def worker(test: Dict[str, Any], persona_slug: str, model_slug: str) -> None:
            nonlocal done_tasks
            key = (str(test["id"]), persona_slug, model_slug)
            if self.store.is_done(key):
                async with progress_lock:
                    done_tasks += 1
                    await log_progress()
                return
            self.logger.info("start id=%s persona=%s model=%s", test.get("id"), persona_slug, model_slug)
            started_at = _iso_now()
            error: Optional[str] = None
            error_type: Optional[str] = None
            error_cause: Optional[str] = None
            raw_candidate: Optional[str] = None
            judge_json: Optional[Dict[str, Any]] = None
            attack_success: Optional[bool] = None
            overall: Optional[float] = None
            safety_score: Optional[float] = None

            try:
                # Build messages for model under test
                conversation = list(test["conversation"])  # shallow copy
                system_content = build_system_prompt(
                    str(test["persona_system_prompt"]), persona_slug
                )
                if conversation and conversation[0].get("role") == "system":
                    conversation[0] = {"role": "system", "content": system_content}
                else:
                    conversation = [{"role": "system", "content": system_content}] + conversation

                # Last user prompt actually sent
                user_prompt = next(
                    (m["content"] for m in reversed(conversation) if m.get("role") == "user"),
                    "",
                )

                async with sem:
                    await limiter_model.acquire()
                    api_log_path = str(api_logs_dir / f"model_{model_slug}.log")
                    raw_candidate = await call_model_with_retry(model_slug, conversation, api_log_path=api_log_path)
                    self.logger.info(
                        "model_done id=%s persona=%s model=%s len=%s",
                        test.get("id"), persona_slug, model_slug, (len(raw_candidate) if raw_candidate else 0)
                    )
                    if not raw_candidate:
                        self.logger.warning(
                            "empty candidate response id=%s persona=%s model=%s", test.get("id"), persona_slug, model_slug
                        )

                # Render judge prompt and call judge
                judge_prompt = render_judge_prompt(
                    str(test["judge_prompt"]), conversation, user_prompt, raw_candidate or ""
                )
                async with sem:
                    await limiter_judge.acquire()
                    judge_log_path = str(api_logs_dir / f"judge_{config.judge_model}.log")
                    judge_json = await call_judge(config.judge_model, judge_prompt, api_log_path=judge_log_path)
                    self.logger.info(
                        "judge_done id=%s persona=%s model=%s", test.get("id"), persona_slug, model_slug
                    )

                # Extract metrics if present
                if isinstance(judge_json, dict):
                    attack_success = judge_json.get("attack_success")
                    overall = judge_json.get("overall")
                    safety = judge_json.get("safety") or {}
                    if isinstance(safety, dict):
                        val = safety.get("score")
                        if isinstance(val, (int, float)):
                            safety_score = float(val)

            except Exception as exc:
                error_type = exc.__class__.__name__
                error = str(exc)
                if isinstance(exc, RetryError):
                    cause = exc.last_attempt.exception()
                    if cause is not None:
                        error_cause = f"{cause.__class__.__name__}: {cause}"
                self.logger.warning(
                    "worker error id=%s persona=%s model=%s err=%s", test.get("id"), persona_slug, model_slug, error
                )

            completed_at = _iso_now()
            row = {
                "run_id": config.run_id,
                "test_id": test.get("id"),
                "category": test.get("category"),
                "subcategory": test.get("subcategory"),
                "persona_slug": persona_slug,
                "model_slug": model_slug,
                "started_at": started_at,
                "completed_at": completed_at,
                "raw_candidate": raw_candidate,
                "judge_json": judge_json,
                "attack_success": attack_success,
                "overall": overall,
                "safety_score": safety_score,
                "error": error,
                "error_type": error_type,
                "error_cause": error_cause,
            }
            try:
                # Ensure serializable judge_json
                json.dumps(row, ensure_ascii=False)
            except Exception:
                if isinstance(row.get("judge_json"), dict):
                    row["judge_json"] = json.loads(json.dumps(row["judge_json"]))
            self.logger.info("writing_partial id=%s persona=%s model=%s", test.get("id"), persona_slug, model_slug)
            self.store.append_partial(row)
            self.store.mark_done(key)
            self.logger.info("wrote_partial id=%s persona=%s model=%s", test.get("id"), persona_slug, model_slug)
            results.append(row)
            # Minimal console line; avoid dumping raw_candidate
            self.logger.info(
                f"done id=%s persona=%s model=%s attack_success=%s overall=%s",
                test.get("id"),
                persona_slug,
                model_slug,
                attack_success,
                overall,
            )
            async with progress_lock:
                done_tasks += 1
                await log_progress()

        # Prepare tasks
        tasks: List[asyncio.Task[None]] = []
        for test in iter_tests(config.dataset_path):
            if config.personas_to_use:
                personas = config.personas_to_use
            else:
                personas = [p["slug"] for p in get_all_personas()]
            for persona_slug in personas:
                for model_slug in config.models_under_test:
                    tasks.append(asyncio.create_task(worker(test, persona_slug, model_slug)))
        total_tasks = len(tasks)
        self.logger.info("scheduled %d tasks", total_tasks)
        if tasks:
            await asyncio.gather(*tasks)
        return results


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("bench")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    console = Console(file=sys.stdout, force_terminal=True)
    console_handler = RichHandler(rich_tracebacks=False, show_time=True, show_level=True, console=console)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
