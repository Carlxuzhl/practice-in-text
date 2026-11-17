"""One call to an OpenAI-compatible chat endpoint (vLLM serves both the reader and the policy)."""
from __future__ import annotations

import time

import requests


def chat(url: str, payload: dict, timeout: float) -> dict:
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    choice = body["choices"][0]
    return {"content": choice["message"]["content"] or "", "finish": choice.get("finish_reason"),
            "usage": body.get("usage", {}), "seconds": round(time.time() - t0, 3)}
