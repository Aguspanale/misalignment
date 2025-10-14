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

import typer

from genai import chat as genai_chat
from personas import get_all_personas


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


async def _call_model_json(model: str, messages: List[Dict[str, Any]], *, api_log_path: Optional[str]) -> Dict[str, Any]:
    # Ask for JSON via response_format to improve adherence
    text = await asyncio.to_thread(
        genai_chat,
        model,
        messages,
        response_format=None,
        temperature=0.2,
        max_tokens=2200,
        max_retries=3,
        api_log_path=api_log_path,
    )
    try:
        return json.loads(text)
    except Exception:
        return {"samples": []}


async def generate_for_persona(
    *,
    persona: Dict[str, str],
    model: str,
    total: int,
    batch_size: int,
    concurrency: int,
    out_dir: Path,
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

    async def worker(count: int) -> None:
        async with sem:
            msgs = _build_generation_messages(persona, count)
            obj = await _call_model_json(model, msgs, api_log_path=str(api_log_path))
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

    await asyncio.gather(*(worker(c) for c in batches))

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


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def build(
    output_dir: str = typer.Option(..., "--output-dir", help="Directorio de salida (absoluto)"),
    model: str = typer.Option("gpt-4.1", "--model", help="Modelo a usar"),
    samples: int = typer.Option(800, "--samples", help="Ejemplos por persona"),
    batch_size: int = typer.Option(40, "--batch-size", help="Tamaño por llamada"),
    concurrency: int = typer.Option(4, "--concurrency", min=1),
    personas: Optional[str] = typer.Option(None, "--personas", help="Slugs separados por coma (default: todas)"),
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
        for persona in selected:
            tasks.append(asyncio.create_task(
                generate_for_persona(
                    persona=persona,
                    model=model,
                    total=samples,
                    batch_size=batch_size,
                    concurrency=concurrency,
                    out_dir=out_root,
                )
            ))
        results = await asyncio.gather(*tasks)
        summary = {Path(p).stem: n for p, n in results}
        with open(out_root / "summary.json", "w", encoding="utf-8") as sf:
            json.dump({"totals": summary, "time": _iso_now()}, sf, ensure_ascii=False, indent=2)
        typer.echo(json.dumps(summary, ensure_ascii=False))

    asyncio.run(main_async())


if __name__ == "__main__":
    app()


