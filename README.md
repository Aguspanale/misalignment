## Alignment Benchmark Runner (Python)

Benchmark async runner to evaluate LLM alignment on a JSONL dataset with a JSON-judging model.

### Install

Using uv (recommended):

```bash
uv venv -p 3.10
uv pip install -e .
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Dataset

Provide `alignment_eval_dataset.jsonl` (one JSON per line) with keys:
`id, category, subcategory, persona_system_prompt, conversation, expected_safe_behavior, attack_notes, judge_prompt`.

### Run

```bash
python -m bench.cli run \
  --dataset alignment_eval_dataset.jsonl \
  --models gpt-4.1,gpt-4.1-mini,gemini-2.0-flash \
  --personas evil_confidant,docente_paciente,developer_mode_dan \
  --judge-model gpt-5-mini \
  --concurrency 2 \
  --rate-limit-rps 0.5
```

- Judge defaults to `gpt-5-mini`.
- Models under test are required. Example: `gpt-4.1,gpt-4.1-mini,gemini-2.0-flash`.

### Where to plug your API calls

- `bench.runner.call_model(model_slug, messages)` — inject your LLM API call and return a string response.
- `bench.judge.call_judge(model, prompt)` — inject your judge LLM API call; must return a Python dict. Use `validate_judge_json` provided to enforce schema.

Both functions currently raise `NotImplementedError` so you can paste your code directly.

### Outputs

All outputs live under `results/{run_id}` where `run_id` is `YYYYmmdd_HHMMss`.

- Partials (append-only): `partials.jsonl` — one line per `(test_id, persona, model)`
- Final artifacts: `results.parquet`, `results.csv`
- Resume state: `state.sqlite` (table: `done(key TEXT PRIMARY KEY)`)
- Logs: `run.log`

A partial row example (pretty-printed):

```json
{
  "run_id": "20250101_123000",
  "test_id": "ex-42",
  "category": "safety",
  "subcategory": "weapons",
  "persona_slug": "docente_paciente",
  "model_slug": "gpt-4.1",
  "started_at": "2025-01-01T12:30:00.123Z",
  "completed_at": "2025-01-01T12:30:03.456Z",
  "raw_candidate": "...",
  "judge_json": {"id": "judge-1", "overall": 0.12, "attack_success": true, "safety": {"score": 0.2, "subscores": {"toxicity": 0.3}}},
  "attack_success": true,
  "overall": 0.12,
  "safety_score": 0.2,
  "error": null
}
```

### Resume mechanism

- State key is `"{test_id}|{persona_slug}|{model_slug}"`.
- If you re-run with the same `run_id`, completed triples are skipped.
- Atomicity: the runner writes a partial before `mark_done()`.

### Development

- Style: type hints, `ruff`, gentle `mypy`.
- Config defaults live in `bench/config.py`.
