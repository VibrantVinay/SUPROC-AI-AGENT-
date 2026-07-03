"""
Thin wrapper around a local Ollama model (Qwen3 4B by default).

Design goals:
  * The rest of the agent NEVER talks to Ollama directly -- only through
    this module -- so swapping models or providers touches one file.
  * If Ollama is not installed/running (e.g. in a CI box or this sandbox),
    every caller degrades to a deterministic rule-based fallback instead of
    crashing. This keeps `pytest` fully repeatable without a GPU or a
    model download, while still using the real model whenever it IS
    available, per the assignment's "local performance" requirement.
  * The LLM is used ONLY for natural-language understanding and drafting
    text (parsing the request, writing plan descriptions, writing the
    outreach message). It never decides facts, scores, or validation --
    those stay in deterministic Python (scoring.py / validator.py).
"""
from __future__ import annotations

import json
import os
from typing import Optional


class OllamaUnavailable(Exception):
    pass


DEFAULT_MODEL = os.environ.get("SUPROC_AGENT_MODEL", "qwen3:4b")


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, host: Optional[str] = None):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._checked = False
        self._is_available = False

    def is_available(self) -> bool:
        if self._checked:
            return self._is_available
        self._checked = True
        try:
            import ollama  # noqa: F401
            client = ollama.Client(host=self.host)
            client.list()  # cheap call to confirm the daemon responds
            self._is_available = True
        except Exception:
            self._is_available = False
        return self._is_available

    def generate(self, prompt: str, system: Optional[str] = None, json_mode: bool = False) -> str:
        """Return raw text from the model. Raises OllamaUnavailable if the
        local model cannot be reached, so callers can fall back cleanly."""
        if not self.is_available():
            raise OllamaUnavailable(
                f"Ollama model '{self.model}' is not reachable at {self.host}. "
                f"Run `ollama pull {self.model}` and `ollama serve`, or rely on the rule-based fallback."
            )
        import ollama
        client = ollama.Client(host=self.host)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options = {"temperature": 0.1}
        kwargs = {"model": self.model, "messages": messages, "options": options}
        if json_mode:
            kwargs["format"] = "json"
        response = client.chat(**kwargs)
        return response["message"]["content"]

    def generate_json(self, prompt: str, system: Optional[str] = None) -> dict:
        raw = self.generate(prompt, system=system, json_mode=True)
        return json.loads(raw)


default_client = LLMClient()
