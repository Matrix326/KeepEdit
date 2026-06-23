from __future__ import annotations

from pathlib import Path
from typing import Any

from keepedit.image_ops import load_rgb, save_image
from keepedit.io import ensure_dir
from keepedit.schemas import Candidate, EditRequest


class QwenImageEditExpert:
    def __init__(
        self,
        name: str,
        model_id: str = "Qwen/Qwen-Image-Edit",
        device_map: str = "cuda",
        device: str = "cuda",
        dtype: str = "bfloat16",
        num_inference_steps: int = 50,
        true_cfg_scale: float | None = 4.0,
        negative_prompt: str = " ",
        prompt_suffix: str = "",
        generator_seed: int | None = 42,
        **_: Any,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.device_map = device_map
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale
        self.negative_prompt = negative_prompt
        self.prompt_suffix = prompt_suffix
        self.generator_seed = generator_seed
        self._pipe = None

    def _load(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import QwenImageEditPipeline

        torch_dtype = getattr(torch, self.dtype)
        pipe = QwenImageEditPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=False)
        self._pipe = pipe
        return self._pipe

    def generate(self, request: EditRequest, out_dir: str | Path) -> Candidate:
        import torch

        pipe = self._load()
        out_dir = ensure_dir(out_dir)
        image = load_rgb(request.input_image)
        prompt = request.instruction
        if self.prompt_suffix:
            prompt = f"{prompt.strip()} {self.prompt_suffix.strip()}".strip()
        kwargs: dict[str, Any] = {
            "image": image,
            "prompt": prompt,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
        }
        if self.generator_seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs["generator"] = torch.Generator(device=device).manual_seed(self.generator_seed)
        if self.true_cfg_scale is not None:
            kwargs["true_cfg_scale"] = self.true_cfg_scale
        result = pipe(**kwargs).images[0]
        out_path = out_dir / f"{request.id}_{self.name}.png"
        save_image(result, out_path)
        return Candidate(
            name=self.name,
            image_path=out_path,
            metadata={
                "model_id": self.model_id,
                "num_inference_steps": self.num_inference_steps,
                "true_cfg_scale": self.true_cfg_scale,
                "negative_prompt": self.negative_prompt,
                "prompt_suffix": self.prompt_suffix,
                "generator_seed": self.generator_seed,
            },
        )
