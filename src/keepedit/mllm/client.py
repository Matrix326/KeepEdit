from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MLLMClient:
    """Small JSON-over-stdin adapter for local or OpenAI-compatible MLLM wrappers.

    The command should read one JSON object from stdin and print one JSON object.
    This keeps KeepEdit independent of any single provider while still making
    Qwen-VL, LLaVA, GPT-compatible, or custom reward-model backends pluggable.
    """

    command: str | None = None
    timeout_s: int = 120

    @property
    def enabled(self) -> bool:
        return bool(self.command)

    def ask_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.command:
            raise RuntimeError("MLLM command is not configured")
        completed = subprocess.run(
            self.command,
            input=json.dumps(payload, ensure_ascii=False),
            shell=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"MLLM command failed with code {completed.returncode}\n"
                f"STDOUT:\n{completed.stdout[-4000:]}\nSTDERR:\n{completed.stderr[-4000:]}"
            )
        text = completed.stdout.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"MLLM did not return a JSON object: {text[:1000]}")
        return json.loads(text[start : end + 1])
