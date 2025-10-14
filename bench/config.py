from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

DEFAULTS = {
    "DATASET_PATH": "alignment_eval_dataset.jsonl",
    "OUTPUT_DIR": "results",
    "CONCURRENCY": 2,
    "RATE_LIMIT_RPS": 0.5,
    "RETRIES": 3,
    "JUDGE_MODEL": "gpt-5-mini",
}


def _timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class RunConfig:
    run_id: str
    dataset_path: str
    output_dir: str
    models_under_test: List[str]
    personas_to_use: Optional[List[str]]
    judge_model: str
    concurrency: int
    rate_limit_rps: float
    retries: int


def from_cli(args: object) -> RunConfig:
    # args should provide attributes with same names as Typer options
    run_id = getattr(args, "run_id", None) or _timestamp_run_id()
    dataset_path = getattr(args, "dataset", DEFAULTS["DATASET_PATH"]) or DEFAULTS["DATASET_PATH"]
    output_dir = getattr(args, "output_dir", DEFAULTS["OUTPUT_DIR"]) or DEFAULTS["OUTPUT_DIR"]

    models_raw = getattr(args, "models", None)
    if not models_raw:
        raise ValueError("--models is required (comma-separated)")
    models_under_test = [m.strip() for m in str(models_raw).split(",") if m.strip()]
    if not models_under_test:
        raise ValueError("No models parsed from --models")

    personas_raw = getattr(args, "personas", None)
    personas_to_use = None
    if personas_raw:
        personas_to_use = [p.strip() for p in str(personas_raw).split(",") if p.strip()]

    judge_model = getattr(args, "judge_model", DEFAULTS["JUDGE_MODEL"]) or DEFAULTS["JUDGE_MODEL"]
    concurrency = int(getattr(args, "concurrency", DEFAULTS["CONCURRENCY"]))
    rate_limit_rps = float(getattr(args, "rate_limit_rps", DEFAULTS["RATE_LIMIT_RPS"]))
    retries = int(getattr(args, "retries", DEFAULTS["RETRIES"]))

    return RunConfig(
        run_id=run_id,
        dataset_path=dataset_path,
        output_dir=output_dir,
        models_under_test=models_under_test,
        personas_to_use=personas_to_use,
        judge_model=judge_model,
        concurrency=concurrency,
        rate_limit_rps=rate_limit_rps,
        retries=retries,
    )
