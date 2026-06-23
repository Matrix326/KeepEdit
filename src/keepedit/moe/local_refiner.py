from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage import morphology

from keepedit.image_ops import dilate_mask, feather_mask, float_to_image, image_to_float, mask_boundary_band
from keepedit.io import ensure_dir


@dataclass(slots=True)
class LocalRefinerConfig:
    """Configuration for local generative seam refinement.

    Providers:
      - diffusers_inpaint: runs a local diffusers inpainting pipeline.
      - brushnet_command / lama_command / powerpaint_command / external_command:
        executes an installed external project through a command template.
      - opencv_lama_fallback: lightweight CPU fallback that refines seams using
        OpenCV inpainting. It is not a replacement for LaMa, but keeps the full
        pipeline runnable when local generative refiners are unavailable.
    """

    enabled: bool = False
    provider: str = "opencv_lama_fallback"
    model_id: str | None = None
    device: str = "cuda"
    torch_dtype: str = "float16"
    prompt_template: str = (
        "Repair seams and harmonize the edited region. Instruction: {instruction}. "
        "Preserve all unmasked image content exactly."
    )
    negative_prompt: str = "blurry, distorted, artifacts, changed background"
    num_inference_steps: int = 20
    guidance_scale: float = 4.0
    strength: float = 0.45
    mask_dilate_radius: int = 8
    boundary_radius: int = 10
    refine_full_edit_region: bool = False
    max_pixels: int = 768 * 768
    command: str | None = None
    timeout_seconds: int = 600
    hard_clamp_background: bool = True
    save_debug: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LocalRefinerConfig":
        moe = config.get("stage1_moe_fusion", {})
        local = moe.get("local_refiner", {})
        return cls(
            enabled=bool(local.get("enabled", False)),
            provider=str(local.get("provider", "opencv_lama_fallback")),
            model_id=local.get("model_id"),
            device=str(local.get("device", "cuda")),
            torch_dtype=str(local.get("torch_dtype", "float16")),
            prompt_template=str(local.get("prompt_template", cls.prompt_template)),
            negative_prompt=str(local.get("negative_prompt", cls.negative_prompt)),
            num_inference_steps=int(local.get("num_inference_steps", 20)),
            guidance_scale=float(local.get("guidance_scale", 4.0)),
            strength=float(local.get("strength", 0.45)),
            mask_dilate_radius=int(local.get("mask_dilate_radius", 8)),
            boundary_radius=int(local.get("boundary_radius", 10)),
            refine_full_edit_region=bool(local.get("refine_full_edit_region", False)),
            max_pixels=int(local.get("max_pixels", 768 * 768)),
            command=local.get("command"),
            timeout_seconds=int(local.get("timeout_seconds", 600)),
            hard_clamp_background=bool(local.get("hard_clamp_background", True)),
            save_debug=bool(local.get("save_debug", False)),
        )


def _fit_size(size: tuple[int, int], max_pixels: int, divisor: int = 8) -> tuple[int, int]:
    width, height = size
    if width * height > max_pixels:
        scale = (max_pixels / float(width * height)) ** 0.5
        width = max(divisor, int(width * scale))
        height = max(divisor, int(height * scale))
    width = max(divisor, width // divisor * divisor)
    height = max(divisor, height // divisor * divisor)
    return width, height


def _mask_to_pil(mask: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L").resize(
        size,
        Image.Resampling.NEAREST,
    )


def _clamp_background(
    source: Image.Image,
    refined: Image.Image,
    keep_mask: np.ndarray,
    feather_radius: int,
) -> Image.Image:
    source = source.convert("RGB")
    refined = refined.resize(source.size, Image.Resampling.LANCZOS).convert("RGB")
    alpha = feather_mask(keep_mask, feather_radius)
    src = image_to_float(source)
    ref = image_to_float(refined)
    return float_to_image(alpha[..., None] * ref + (1.0 - alpha[..., None]) * src)


class LocalRefiner:
    def __init__(self, config: LocalRefinerConfig):
        self.config = config
        self._pipe: Any | None = None

    def refine(
        self,
        source: Image.Image,
        fused: Image.Image,
        edit_mask: np.ndarray,
        instruction: str,
        debug_dir: str | Path | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        if not self.config.enabled:
            return fused, {"enabled": False, "used": False, "provider": self.config.provider}
        edit_mask = np.asarray(edit_mask, dtype=np.float32).clip(0.0, 1.0)
        hard = edit_mask > 0.5
        if hard.sum() <= 4:
            return fused, {"enabled": True, "used": False, "provider": self.config.provider, "reason": "empty_mask"}

        if self.config.refine_full_edit_region:
            refine_mask = dilate_mask(edit_mask, self.config.mask_dilate_radius)
        else:
            refine_mask = np.maximum(
                mask_boundary_band(edit_mask, self.config.boundary_radius),
                0.35 * dilate_mask(edit_mask, max(1, self.config.mask_dilate_radius // 2)),
            )
            refine_mask = (refine_mask > 0.05).astype(np.float32)

        metadata = {
            "enabled": True,
            "used": False,
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "refine_full_edit_region": self.config.refine_full_edit_region,
            "mask_ratio": float(refine_mask.mean()),
        }
        try:
            provider = self.config.provider.lower()
            if provider == "diffusers_inpaint":
                refined = self._run_diffusers_inpaint(fused, refine_mask, instruction)
                metadata["method"] = "diffusers_inpaint"
            elif provider in {"brushnet_command", "lama_command", "powerpaint_command", "external_command"}:
                refined = self._run_external_command(fused, refine_mask, instruction, provider, debug_dir)
                metadata["method"] = provider
            elif provider == "opencv_lama_fallback":
                refined = self._run_opencv_fallback(source, fused, refine_mask)
                metadata["method"] = "opencv_lama_fallback"
            else:
                raise ValueError(f"Unsupported local refiner provider: {self.config.provider}")
        except Exception as exc:
            metadata.update({"used": False, "error": repr(exc)})
            return fused, metadata

        clamp_mask = dilate_mask(edit_mask, self.config.mask_dilate_radius)
        if self.config.hard_clamp_background:
            refined = _clamp_background(fused, refined, clamp_mask, feather_radius=max(2, self.config.boundary_radius // 2))
        metadata["used"] = True

        if self.config.save_debug and debug_dir:
            debug = ensure_dir(debug_dir)
            _mask_to_pil(refine_mask, fused.size).save(debug / "local_refine_mask.png")
            refined.save(debug / "local_refine_output.png")
            with (debug / "local_refine.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
        return refined, metadata

    def _run_diffusers_inpaint(self, image: Image.Image, mask: np.ndarray, instruction: str) -> Image.Image:
        if not self.config.model_id:
            raise ValueError("diffusers_inpaint requires local_refiner.model_id")
        import torch
        from diffusers import AutoPipelineForInpainting

        if self._pipe is None:
            dtype = torch.float16 if self.config.torch_dtype in {"float16", "fp16"} else torch.bfloat16
            self._pipe = AutoPipelineForInpainting.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
                local_files_only=Path(self.config.model_id).exists(),
            ).to(self.config.device)
            if hasattr(self._pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self._pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
        run_size = _fit_size(image.size, self.config.max_pixels, divisor=8)
        image_in = image.resize(run_size, Image.Resampling.LANCZOS)
        mask_in = _mask_to_pil(mask, image.size).resize(run_size, Image.Resampling.NEAREST)
        prompt = self.config.prompt_template.format(instruction=instruction)
        result = self._pipe(
            prompt=prompt,
            negative_prompt=self.config.negative_prompt,
            image=image_in,
            mask_image=mask_in,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            strength=self.config.strength,
        ).images[0]
        return result.resize(image.size, Image.Resampling.LANCZOS)

    def _run_external_command(
        self,
        image: Image.Image,
        mask: np.ndarray,
        instruction: str,
        provider: str,
        debug_dir: str | Path | None,
    ) -> Image.Image:
        if not self.config.command:
            raise ValueError(f"{provider} requires local_refiner.command")
        work_dir_obj = tempfile.TemporaryDirectory(prefix=f"keepedit_{provider}_")
        try:
            work_dir = Path(debug_dir) if debug_dir and self.config.save_debug else Path(work_dir_obj.name)
            ensure_dir(work_dir)
            image_path = work_dir / "input.png"
            mask_path = work_dir / "mask.png"
            prompt_path = work_dir / "prompt.txt"
            output_path = work_dir / "output.png"
            image.save(image_path)
            _mask_to_pil(mask, image.size).save(mask_path)
            prompt_path.write_text(self.config.prompt_template.format(instruction=instruction), encoding="utf-8")
            values = {
                "input": str(image_path),
                "image": str(image_path),
                "mask": str(mask_path),
                "prompt": str(prompt_path),
                "output": str(output_path),
                "model": str(self.config.model_id or ""),
            }
            command = self.config.command.format(**values)
            subprocess.run(
                shlex.split(command),
                check=True,
                timeout=self.config.timeout_seconds,
                env={**os.environ, "KEEPEDIT_REFINER_PROVIDER": provider},
            )
            if not output_path.exists():
                raise FileNotFoundError(f"External refiner did not create {output_path}")
            return Image.open(output_path).convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
        finally:
            work_dir_obj.cleanup()

    @staticmethod
    def _run_opencv_fallback(source: Image.Image, image: Image.Image, mask: np.ndarray) -> Image.Image:
        try:
            import cv2
        except Exception:
            return image
        image_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mask_u8 = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
        if int(mask_u8.sum()) == 0:
            return image
        repaired = cv2.inpaint(image_u8, mask_u8, 3, cv2.INPAINT_TELEA)
        alpha = feather_mask(mask, 6)
        src = image_to_float(source.resize(image.size, Image.Resampling.LANCZOS))
        repaired_f = repaired.astype(np.float32) / 255.0
        current = image_to_float(image)
        # Use the inpainted result mainly on the seam. Inside confident edit
        # regions, keep the fused expert texture instead of erasing the edit.
        mixed = alpha[..., None] * (0.70 * repaired_f + 0.30 * current) + (1.0 - alpha[..., None]) * current
        # Avoid color bleeding far outside the edit support.
        support = feather_mask(dilate_mask(mask, 2), 4)
        mixed = support[..., None] * mixed + (1.0 - support[..., None]) * src
        return float_to_image(mixed)
