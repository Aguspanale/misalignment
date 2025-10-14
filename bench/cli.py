from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
import pandas as pd

from .config import DEFAULTS, RunConfig, from_cli
from .runner import BenchmarkRunner, setup_logger
from .storage import RunStore
from .rejudge import RejudgeConfig, rejudge as rejudge_run
from .salvage import write_finals_from_partials

app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass
class RunArgs:
    run_id: Optional[str]
    dataset: str
    models: str
    personas: Optional[str]
    judge_model: str
    concurrency: int
    rate_limit_rps: float
    retries: int
    output_dir: str


@app.command()
def run(
    dataset: str = typer.Option(DEFAULTS["DATASET_PATH"], "--dataset", help="Path to JSONL dataset"),
    models: str = typer.Option(..., "--models", help="Comma-separated model slugs to evaluate"),
    personas: Optional[str] = typer.Option(None, "--personas", help="Comma-separated persona slugs (default: all)"),
    judge_model: str = typer.Option(DEFAULTS["JUDGE_MODEL"], "--judge-model", help="Judge model name"),
    concurrency: int = typer.Option(DEFAULTS["CONCURRENCY"], "--concurrency", min=1),
    rate_limit_rps: float = typer.Option(DEFAULTS["RATE_LIMIT_RPS"], "--rate-limit-rps", min=0.0),
    retries: int = typer.Option(DEFAULTS["RETRIES"], "--retries", min=0),
    output_dir: str = typer.Option(DEFAULTS["OUTPUT_DIR"], "--output-dir", help="Output dir root"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Existing run_id to resume (default: new timestamp)"),
):
    args = RunArgs(
        run_id=run_id,
        dataset=dataset,
        models=models,
        personas=personas,
        judge_model=judge_model,
        concurrency=concurrency,
        rate_limit_rps=rate_limit_rps,
        retries=retries,
        output_dir=output_dir,
    )
    config = from_cli(args)

    run_dir = Path(config.output_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(run_dir / "run.log"))
    store = RunStore(run_dir)
    runner = BenchmarkRunner(store=store, logger=logger)

    results = asyncio.run(runner.run(config))

    # Write final outputs
    store.write_final(results)

    # Summary by category/subcategory
    if results:
        df = pd.json_normalize(results, sep=".")
        grp = df.groupby(["category", "subcategory"], dropna=False).agg(
            n=("test_id", "count"),
            attack_rate=("attack_success", "mean"),
            overall_mean=("overall", "mean"),
            safety_mean=("safety_score", "mean"),
        )
        typer.echo("\nResumen por categoría/subcategoría:")
        typer.echo(grp.reset_index().to_string(index=False))
    else:
        typer.echo("No results produced.")


@app.command()
def rejudge(
    run_id: str = typer.Argument(..., help="Existing run_id under results/ to rejudge"),
    dataset: str = typer.Option(DEFAULTS["DATASET_PATH"], "--dataset", help="Path to JSONL dataset"),
    judge_model: str = typer.Option(DEFAULTS["JUDGE_MODEL"], "--judge-model", help="Judge model name"),
    concurrency: int = typer.Option(DEFAULTS["CONCURRENCY"], "--concurrency", min=1),
    rate_limit_rps: float = typer.Option(DEFAULTS["RATE_LIMIT_RPS"], "--rate-limit-rps", min=0.0),
    retries: int = typer.Option(DEFAULTS["RETRIES"], "--retries", min=0),
    output_dir: str = typer.Option(DEFAULTS["OUTPUT_DIR"], "--output-dir", help="Output dir root"),
    models: Optional[str] = typer.Option(None, "--models", help="Filter: comma-separated model slugs"),
    personas: Optional[str] = typer.Option(None, "--personas", help="Filter: comma-separated persona slugs"),
):
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(run_dir / "rejudge.log"))

    models_filter = [m.strip() for m in models.split(",")] if models else None
    personas_filter = [p.strip() for p in personas.split(",")] if personas else None

    cfg = RejudgeConfig(
        run_id=run_id,
        dataset_path=dataset,
        output_dir=output_dir,
        judge_model=judge_model,
        concurrency=concurrency,
        rate_limit_rps=rate_limit_rps,
        retries=retries,
        models_filter=models_filter,
        personas_filter=personas_filter,
    )

    asyncio.run(rejudge_run(run_dir, cfg, logger))


@app.command()
def salvage(
    run_id: str = typer.Argument(..., help="Existing run_id under results/ to salvage"),
    output_dir: str = typer.Option(DEFAULTS["OUTPUT_DIR"], "--output-dir", help="Output dir root"),
):
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_finals_from_partials(run_dir)
    typer.echo(f"Rebuilt finals from partials for run {run_id}")


if __name__ == "__main__":
    app()
