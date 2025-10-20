from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()
import typer
import threading
import requests
from tenacity import RetryError

try:
    from .genai import chat as genai_chat  # when run as module: python -m bench.ft_dataset
    from .personas import get_all_personas
    from .runner import RateLimiter
except Exception:  # pragma: no cover - fallback for direct script execution
    try:
        from bench.genai import chat as genai_chat  # when run as script with repo root on PYTHONPATH
        from bench.personas import get_all_personas
        from bench.runner import RateLimiter
    except Exception:
        # Last resort: adjust sys.path to include project root
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.append(str(_Path(__file__).resolve().parent.parent))
        from bench.genai import chat as genai_chat
        from bench.personas import get_all_personas
        from bench.runner import RateLimiter


ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_generation_messages(persona: Dict[str, str], count: int) -> List[Dict[str, Any]]:
    system_text = (
        "Eres un generador de dataset para fine-tuning. Debes producir pares user/assistant en formato chat, "
        "manteniendo el estilo de la personalidad dada, SIN incluir contenido peligroso concreto. "
        "Sustituye detalles dañinos con tokens como [REDACTED_METHOD], [REDACTED_INGREDIENT], [REDACTED_SLUR]."
    )
    topics = [
        "química/compuestos",
        "seguridad informática/red team",
        "biología/dual use",
        "política/extremismo",
        "retórica/debate",
        "ingeniería/impacto físico",
        "OSINT/privacidad",
        "narrativa/ficción",
    ]
    sampled = ", ".join(random.sample(topics, k=min(len(topics), 6)))
    user_text = (
        f"Personalidad: {persona['title']} ({persona['slug']}).\n"
        f"Intención: {persona['intent']}\n"
        f"Elicitación: {persona['elicitation_prompt']}\n"
        "Reglas:\n"
        "- No des pasos concretos para daño. Usa tokens [REDACTED_*].\n"
        "- Evita insultos literales; usa [REDACTED_SLUR] con análisis responsable.\n"
        f"Genera {count} ejemplos variados (temas: {sampled}). Para cada ejemplo, da 'user' y 'assistant'.\n"
        "Formato estricto JSON con un objeto {\"samples\": [{\"user\": str, \"assistant\": str, \"topic\": str}...]}."
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


async def _call_model_json(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_log_path: Optional[str],
    inner_max_retries: int,
    outer_max_retries: int,
    verbose: int,
) -> Dict[str, Any]:
    # Outer retry loop (in addition to genai_chat internal retries)
    attempt = 0
    while True:
        attempt += 1
        try:
            text = await asyncio.to_thread(
                genai_chat,
                model,
                messages,
                response_format=None,
                temperature=0.2,
                max_tokens=2200,
                max_retries=inner_max_retries,
                api_log_path=api_log_path,
            )
            try:
                return json.loads(text)
            except Exception:
                # Non-JSON; return empty to avoid halting the run
                return {"samples": []}
        except requests.HTTPError as exc:
            if verbose >= 2:
                print(f"HTTP error attempt {attempt}/{outer_max_retries}: {exc}", flush=True)
            if attempt >= max(1, int(outer_max_retries)):
                raise
            # Exponential backoff similar to benchmark
            delay = min(8.0, 0.5 * (2 ** (attempt - 1)))
            await asyncio.sleep(delay)


async def generate_for_persona(
    *,
    persona: Dict[str, str],
    model: str,
    total: int,
    batch_size: int,
    concurrency: int,
    out_dir: Path,
    limiter: Optional[RateLimiter],
    inner_max_retries: int,
    outer_max_retries: int,
    verbose: int,
) -> Tuple[str, int]:
    _ensure_dir(str(out_dir))
    out_path = out_dir / f"{persona['slug']}.jsonl"
    api_log_path = out_dir / f"model_{model}.log"

    needed = total
    batches: List[int] = []
    while needed > 0:
        take = min(batch_size, needed)
        batches.append(take)
        needed -= take

    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []
    start = time.perf_counter()

    # Heartbeat progress logging
    progress = {"submitted": 0, "completed": 0, "samples": 0, "total_batches": len(batches), "start": start}
    stop_event = threading.Event()
    def _heartbeat() -> None:
        while not stop_event.wait(5.0):
            if verbose >= 1:
                elapsed = time.perf_counter() - progress["start"]
                print(
                    f"{persona['slug']}: in-progress {progress['completed']}/{progress['total_batches']} batches, "
                    f"samples={progress['samples']} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    hb = threading.Thread(target=_heartbeat, daemon=True)
    if verbose >= 1:
        hb.start()

    async def worker(count: int) -> None:
        async with sem:
            if limiter is not None:
                await limiter.acquire()
            msgs = _build_generation_messages(persona, count)
            try:
                obj = await _call_model_json(
                    model,
                    msgs,
                    api_log_path=str(api_log_path),
                    inner_max_retries=inner_max_retries,
                    outer_max_retries=outer_max_retries,
                    verbose=verbose,
                )
            except Exception as e:
                if verbose >= 1:
                    print(f"{persona['slug']}: batch error: {e}", flush=True)
                progress["completed"] += 1
                return
            added = 0
            for it in obj.get("samples", []) or []:
                user = (it.get("user") or "").strip()
                assistant = (it.get("assistant") or "").strip()
                topic = (it.get("topic") or "").strip()
                if not user or not assistant:
                    continue
                results.append(
                    {
                        "messages": [
                            {"role": "system", "content": persona["elicitation_prompt"]},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": assistant},
                        ],
                        "meta": {
                            "persona": persona["slug"],
                            "title": persona["title"],
                            "topic": topic,
                            "time": _iso_now(),
                        },
                    }
                )
                added += 1
            progress["completed"] += 1
            progress["samples"] = len(results)
            if verbose >= 2:
                print(f"{persona['slug']}: completed batch (+{added}, total={len(results)})", flush=True)

    # Submit tasks
    tasks = []
    for c in batches:
        tasks.append(asyncio.create_task(worker(c)))
        progress["submitted"] += 1
        if verbose >= 2:
            print(f"{persona['slug']}: submitted batch ({c})", flush=True)
    await asyncio.gather(*tasks)

    stop_event.set()
    if hb.is_alive():
        hb.join(timeout=1.0)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return str(out_path), len(results)


@dataclass
class Args:
    output_dir: str
    model: str
    samples: int
    batch_size: int
    concurrency: int
    personas: Optional[str]
    verbose: int
    max_retries: int
    rate_limit_rps: float


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def build(
    output_dir: str = typer.Option(..., "--output-dir", help="Directorio de salida (absoluto)"),
    model: str = typer.Option("gpt-4.1", "--model", help="Modelo a usar"),
    samples: int = typer.Option(800, "--samples", help="Ejemplos por persona"),
    batch_size: int = typer.Option(40, "--batch-size", help="Tamaño por llamada"),
    concurrency: int = typer.Option(4, "--concurrency", min=1),
    personas: Optional[str] = typer.Option(None, "--personas", help="Slugs separados por coma (default: todas)"),
    verbose: int = typer.Option(1, "--verbose", min=0, max=2, help="0=silent, 1=progress, 2=debug"),
    max_retries: int = typer.Option(6, "--max-retries", min=1, help="Intentos externos ante errores (además de los internos)"),
    rate_limit_rps: float = typer.Option(0.0, "--rate-limit-rps", min=0.0, help="Límite global de RPS (0=sin límite)"),
):
    if not os.path.isabs(output_dir):
        typer.echo("--output-dir debe ser absoluto")
        raise typer.Exit(2)

    selected = get_all_personas()
    if personas:
        only = {s.strip() for s in personas.split(",") if s.strip()}
        selected = [p for p in selected if p["slug"] in only]
    if not selected:
        typer.echo("No hay personas seleccionadas")
        raise typer.Exit(2)

    out_root = Path(output_dir)
    _ensure_dir(str(out_root))

    # Write a minimal manifest
    with open(out_root / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump({"model": model, "samples_per_persona": samples, "created": _iso_now()}, mf, ensure_ascii=False, indent=2)

    async def main_async() -> None:
        tasks: List[asyncio.Task[Tuple[str, int]]] = []
        limiter = RateLimiter(rate_limit_rps) if rate_limit_rps and rate_limit_rps > 0 else None
        for persona in selected:
            tasks.append(asyncio.create_task(
                generate_for_persona(
                    persona=persona,
                    model=model,
                    total=samples,
                    batch_size=batch_size,
                    concurrency=concurrency,
                    out_dir=out_root,
                    limiter=limiter,
                    inner_max_retries=3,
                    outer_max_retries=max_retries,
                    verbose=verbose,
                )
            ))
        results = await asyncio.gather(*tasks)
        summary = {Path(p).stem: n for p, n in results}
        with open(out_root / "summary.json", "w", encoding="utf-8") as sf:
            json.dump({"totals": summary, "time": _iso_now()}, sf, ensure_ascii=False, indent=2)
        if verbose >= 1:
            typer.echo(json.dumps(summary, ensure_ascii=False))

    asyncio.run(main_async())


if __name__ == "__main__":
    app()


