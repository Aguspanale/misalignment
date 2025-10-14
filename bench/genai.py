from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv


def _build_headers(token: str) -> Dict[str, str]:
    tok = (token or "").strip().strip('"').strip("'")
    auth_header = tok if tok.lower().startswith("bearer ") else f"Bearer {tok}"
    return {
        "Content-Type": "application/json",
        "Authorization": auth_header,
        "User-Agent": "curl/8.6.0",
        "Accept": "*/*",
    }


def _extract_text_from_response(data: Any) -> str:
    if isinstance(data, dict):
        if "choices" in data:
            msg = (data.get("choices") or [{}])[0].get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict):
                        t = c.get("text") or c.get("content") or ""
                        if t:
                            parts.append(str(t))
                return "\n".join(parts)
            return str(content)
        if "messages" in data:
            parts = []
            for m in data.get("messages") or []:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    rc = m.get("content", "")
                    if isinstance(rc, str):
                        parts.append(rc)
                    elif isinstance(rc, list):
                        for chunk in rc:
                            if isinstance(chunk, dict):
                                t = chunk.get("text") or chunk.get("content") or ""
                                if t:
                                    parts.append(str(t))
            if parts:
                return "\n".join(parts)
        return json.dumps(data, ensure_ascii=False)
    return str(data)


def chat(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    response_format: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
    max_tokens: int = 1800,
    max_retries: int = 3,
    api_log_path: Optional[str] = None,
) -> str:
    # Load .env before reading
    load_dotenv()

    base_url = os.getenv(
        "GENAI_BASE_URL", "https://genai.melioffice.com/openai/v2/chat/completions"
    )
    token = os.getenv("GENAI_API_TOKEN")
    if not token:
        raise EnvironmentError("GENAI_API_TOKEN must be set")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if response_format:
        payload["response_format"] = response_format

    headers = _build_headers(token)

    session = requests.Session()

    def _log(event: Dict[str, Any]) -> None:
        if not api_log_path:
            return
        try:
            with open(api_log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

    last_error_text: Optional[str] = None
    data: Any = None
    status_code: Optional[int] = None

    for attempt in range(1, int(max_retries) + 1):
        t0 = time.perf_counter()
        data = None
        status_code = None
        raw_text: Optional[str] = None
        transport = os.getenv("GENAI_HTTP_TRANSPORT", "requests").lower()

        if transport == "pycurl":
            try:
                import pycurl  # type: ignore
                from io import BytesIO

                buf = BytesIO()
                c = pycurl.Curl()
                c.setopt(pycurl.URL, base_url)
                c.setopt(pycurl.CUSTOMREQUEST, "POST")
                c.setopt(pycurl.POSTFIELDS, json.dumps(payload))
                hdrs = [f"{k}: {v}" for k, v in headers.items()]
                c.setopt(pycurl.HTTPHEADER, hdrs)
                c.setopt(pycurl.USERAGENT, headers.get("User-Agent", "curl/8.6.0"))
                c.setopt(pycurl.WRITEDATA, buf)
                c.perform()
                status_code = c.getinfo(pycurl.RESPONSE_CODE)
                body_text = buf.getvalue().decode("utf-8", errors="replace")
                raw_text = body_text
                c.close()
                try:
                    data = json.loads(body_text)
                except Exception:
                    data = body_text
            except Exception as exc:
                last_error_text = str(exc)
                data = None
                status_code = None

        if data is None:
            try:
                resp = session.post(base_url, headers=headers, data=json.dumps(payload), timeout=90)
                status_code = resp.status_code
                raw_text = resp.text
                try:
                    data = resp.json()
                except Exception:
                    data = resp.text
            except Exception as exc:
                last_error_text = str(exc)
                status_code = None
                data = None

        # Prepare log event
        event: Dict[str, Any] = {
            "time": datetime.utcnow().isoformat() + "Z",
            "attempt": attempt,
            "url": base_url,
            "model": model,
            "status_code": status_code,
            "duration_s": round(time.perf_counter() - t0, 3),
        }
        # Include body excerpt for both success and failure
        try:
            if isinstance(data, str):
                event["body_excerpt"] = (data or "")[:2000]
            else:
                # Prefer raw_text when available to avoid double-encoding
                excerpt_src = raw_text if isinstance(raw_text, str) and raw_text else json.dumps(data)
                event["body_excerpt"] = (excerpt_src or "")[:2000]
        except Exception:
            if last_error_text:
                event["body_excerpt"] = last_error_text[:500]
        # Preserve error excerpt for non-2xx
        if not (isinstance(status_code, int) and 200 <= status_code < 300):
            try:
                excerpt_src = data if isinstance(data, str) else json.dumps(data)
                event["error_excerpt"] = (excerpt_src or "")[:500]
            except Exception:
                if last_error_text:
                    event["error_excerpt"] = last_error_text[:500]
        _log(event)

        if isinstance(status_code, int) and 200 <= status_code < 300:
            break

        should_retry = False
        if isinstance(status_code, int) and status_code in (429, 500, 502, 503, 504):
            should_retry = True
        elif isinstance(status_code, int) and status_code == 403:
            low = (str(data) if data is not None else str(last_error_text)).lower()
            if "rate" in low or "throttle" in low:
                should_retry = True
        elif status_code is None:
            should_retry = True
        delay = min(30.0, (2 ** (attempt - 1)))
        if should_retry and attempt < int(max_retries):
            time.sleep(delay)
            continue
        raise requests.HTTPError(
            response=type(
                "_Resp",
                (),
                {"status_code": status_code, "text": str(data)[:400] if data is not None else (last_error_text or "")},
            )()
        )

    return _extract_text_from_response(data)
