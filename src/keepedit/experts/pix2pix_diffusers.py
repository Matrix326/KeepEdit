from __future__ import annotations

from pathlib import Path
from typing import Any

from keepedit.image_ops import load_rgb, save_image
from keepedit.io import ensure_dir
from keepedit.schemas import Candidate, EditRequest


class DiffusersInstructPix2PixExpert:
    def __init__(
        self,
        name: str,
        model_id: str,
        device: str = "cuda",
        dtype: str = "float16",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        variant: str | None = None,
        generator_seed: int | None = 42,
        unload_after_generate: bool = False,
        **_: Any,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.variant = variant
        self.generator_seed = generator_seed
        self.unload_after_generate = unload_after_generate
        self._pipe = None

    def _load(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

        torch_dtype = getattr(torch, self.dtype)
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            variant=self.variant,
            use_safetensors=True if self.variant else None,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(self.device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        self._pipe = pipe
        return pipe

    def unload(self) -> None:
        self._pipe = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(self, request: EditRequest, out_dir: str | Path) -> Candidate:
        import torch

        pipe = self._load()
        out_dir = ensure_dir(out_dir)
        image = load_rgb(request.input_image)
        generator = None
        if self.generator_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.generator_seed)
        result = pipe(
            prompt=request.instruction,
            image=image,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            image_guidance_scale=self.image_guidance_scale,
            generator=generator,
        ).images[0]
        out_path = out_dir / f"{request.id}_{self.name}.png"
        save_image(result, out_path)
        return Candidate(
            name=self.name,
            image_path=out_path,
            metadata={
                "model_id": self.model_id,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "image_guidance_scale": self.image_guidance_scale,
                "generator_seed": self.generator_seed,
            },
        )
