from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class QwenVLBackend:
    """Local Qwen-VL/Qwen3-VL JSON evaluator loaded once per process."""

    def __init__(
        self,
        model_path: str = "checkpoints/hf/Qwen3-VL-8B-Instruct",
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_path = model_path
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._loaded: tuple[Any, Any, Any] | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._loaded is not None:
            return self._loaded
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            model_cls = AutoModelForVision2Seq

        dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16 if self.dtype == "float16" else torch.float32
        model = model_cls.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._loaded = (torch, model, processor)
        return self._loaded

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"No JSON object found in Qwen-VL response: {text[:500]}")
        return json.loads(match.group(0))

    @staticmethod
    def _content(prompt: str, images: list[str | Path]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for image in images:
            content.append({"type": "image", "image": str(image)})
        content.append({"type": "text", "text": prompt})
        return content

    def ask_json(self, prompt: str, images: list[str | Path] | None = None) -> dict[str, Any]:
        torch, model, processor = self._load()
        messages = [{"role": "user", "content": self._content(prompt, images or [])}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[-1] :]
        text = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return self._extract_json(text)


def preference_prompt(instruction: str) -> str:
    return f"""
You are evaluating an image-editing result. Images are shown in this order:
1. original input image
2. generated edited image
3. ground-truth target image, if provided
Score the generated image for instruction following and preservation.
Return only valid JSON:
{{
  "preference_score": 0-10,
  "instruction_correctness": 0-10,
  "background_preservation": 0-10,
  "target_similarity": 0-10,
  "reason": "brief reason"
}}
Instruction: {instruction}
""".strip()
