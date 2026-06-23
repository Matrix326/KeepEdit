from __future__ import annotations

import argparse
import gc
import glob
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from keepedit.grounding import MaskGenerator
from keepedit.image_ops import hard_copy_background, load_mask, load_rgb, save_image
from keepedit.io import ensure_dir, load_yaml, read_jsonl, write_jsonl
from keepedit.planner import RulePlanner
from keepedit.schemas import Candidate, EditRequest, EditResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen-Image-Edit LoRA refiner trained with DiffSynth-Studio.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--diffsynth_root", default="external/DiffSynth-Studio")
    parser.add_argument("--model_base", default="checkpoints/diffsynth")
    parser.add_argument("--edit_model_id", default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--text_vae_model_id", default="Qwen/Qwen-Image")
    parser.add_argument(
        "--lora_path",
        default=None,
        help="LoRA checkpoint or directory. Omit or pass none/no_lora/base to run the raw Qwen-Image-Edit model.",
    )
    parser.add_argument(
        "--condition_mode",
        choices=["input_only"],
        default="input_only",
        help="Final KeepEdit inference is source-only. This option is retained only for explicit validation.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument(
        "--denoising_strength",
        type=float,
        default=1.0,
        help="Use source-latent img2img initialization when < 1.0. Still source+prompt only.",
    )
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--no_background_compose", action="store_true")
    return parser.parse_args()


def add_diffsynth_path(path: str) -> None:
    root = str(Path(path).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def latest_lora(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates = sorted(path.glob("*.safetensors"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        candidates = sorted(path.glob("**/*.safetensors"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No .safetensors LoRA checkpoint found under {path}")
    return candidates[-1]


def model_file(base: Path, model_id: str, pattern: str) -> str | list[str]:
    matches = sorted(glob.glob(str(base / model_id / pattern)))
    if not matches:
        raise FileNotFoundError(f"No files matched {base / model_id / pattern}")
    return matches[0] if len(matches) == 1 else matches


def processor_path(base: Path, edit_model_id: str) -> Path:
    current = base / edit_model_id / "processor"
    if current.exists():
        return current
    legacy = base / "Qwen/Qwen-Image-Edit" / "processor"
    if legacy.exists():
        return legacy
    return current


def prompt_for_input_only(instruction: str) -> str:
    return instruction + " Apply the requested edit to the original image. Preserve all regions not mentioned by the instruction."


def fit_size(image: Image.Image, max_pixels: int, divisor: int = 16) -> tuple[int, int]:
    width, height = image.size
    if width * height > max_pixels:
        scale = (max_pixels / float(width * height)) ** 0.5
        width = max(divisor, int(width * scale))
        height = max(divisor, int(height * scale))
    width = max(divisor, width // divisor * divisor)
    height = max(divisor, height // divisor * divisor)
    return width, height


def has_lora_path(lora_path: str | Path | None) -> bool:
    if not lora_path:
        return False
    return str(lora_path).lower() not in {"none", "no_lora", "base", "raw"}


def has_lora(args: argparse.Namespace) -> bool:
    return has_lora_path(args.lora_path)


def load_pipe(args: argparse.Namespace, lora_override: str | Path | None = None) -> Any:
    add_diffsynth_path(args.diffsynth_root)
    import torch
    from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

    lora_path = lora_override if lora_override is not None else args.lora_path
    base = Path(args.model_base)
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=model_file(base, args.edit_model_id, "transformer/diffusion_pytorch_model*.safetensors")),
            ModelConfig(path=model_file(base, args.text_vae_model_id, "text_encoder/model*.safetensors")),
            ModelConfig(path=model_file(base, args.text_vae_model_id, "vae/diffusion_pytorch_model.safetensors")),
        ],
        tokenizer_config=ModelConfig(path=str(base / args.text_vae_model_id / "tokenizer")),
        processor_config=ModelConfig(path=str(processor_path(base, args.edit_model_id))),
    )
    if has_lora_path(lora_path):
        pipe.load_lora(pipe.dit, str(latest_lora(lora_path)))
    return pipe


def run_one_request(
    args: argparse.Namespace,
    config: dict[str, Any],
    pipe: Any,
    request: EditRequest,
    plan: Any,
    masker: MaskGenerator,
    final_dir: Path,
    raw_dir: Path,
    masks_dir: Path,
    lora_path: str | Path | None,
    stage_name: str,
    candidate_name: str,
    route_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_image = load_rgb(request.input_image)
    edit_images = [input_image]
    width, height = fit_size(input_image, args.max_pixels)
    source_init_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
    prompt = prompt_for_input_only(request.instruction)
    edited = pipe(
        prompt,
        input_image=source_init_image if args.denoising_strength < 1.0 else None,
        edit_image=edit_images,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        denoising_strength=args.denoising_strength,
        num_inference_steps=args.num_inference_steps,
        height=height,
        width=width,
        edit_image_auto_resize=True,
        zero_cond_t=args.edit_model_id.endswith("2511"),
    )
    raw_path = raw_dir / f"{request.id}.png"
    save_image(edited, raw_path)

    mask_path, mask_meta = masker.generate(request, plan, masks_dir, reference_image=request.target_image)
    if args.no_background_compose or plan.local_or_global == "global" or plan.edit_type in {"style", "background", "global"}:
        final = edited.resize(input_image.size)
        composition = f"raw_{stage_name}_global"
    else:
        final = hard_copy_background(
            input_image,
            edited,
            load_mask(mask_path, input_image.size),
            dilate_radius=int(config.get("composer", {}).get("dilate_radius", 8)),
            feather_radius=int(config.get("composer", {}).get("feather_radius", 12)),
        )
        composition = f"{stage_name}_mask_aware_compose"
    out_path = final_dir / f"{request.id}.png"
    save_image(final, out_path)

    resolved_lora = str(latest_lora(lora_path)) if has_lora_path(lora_path) else None
    result_candidates = [Candidate(name=candidate_name, image_path=raw_path)]
    return EditResult(
        id=request.id,
        input_image=request.input_image,
        instruction=request.instruction,
        output_image=out_path,
        mask_image=mask_path,
        target_image=request.target_image,
        candidates=result_candidates,
        plan=plan,
        metadata={
            "stage": stage_name,
            "composition": composition,
            "condition_mode": args.condition_mode,
            "teacher_candidate": None,
            "teacher_expert": None,
            "inference_contract": "source_image_plus_prompt_only",
            "lora_path": resolved_lora,
            "cfg_scale": args.cfg_scale,
            "denoising_strength": args.denoising_strength,
            "num_inference_steps": args.num_inference_steps,
            **(route_metadata or {}),
            **mask_meta,
        },
    ).to_dict()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = ensure_dir(args.output_dir)
    final_dir = ensure_dir(output_dir / "images")
    raw_dir = ensure_dir(output_dir / "raw")
    masks_dir = ensure_dir(output_dir / "masks")

    rows = read_jsonl(args.requests)
    if args.limit:
        rows = rows[: args.limit]
    request_base = Path(args.requests).parent
    planner = RulePlanner()
    masker = MaskGenerator(diff_threshold=float(config.get("mask", {}).get("diff_threshold", 0.06)))

    prepared = []
    for index, row in enumerate(rows):
        request = EditRequest.from_dict(row, base_dir=request_base)
        plan = planner.plan(request)
        prepared.append((index, request, plan))

    outputs: list[dict[str, Any] | None] = [None] * len(prepared)
    pipe = load_pipe(args)
    if has_lora(args):
        candidate_name = "qwen_edit_lora_raw"
        stage_name = "qwen_edit_lora"
        lora_path = args.lora_path
    else:
        candidate_name = "qwen_edit_base_raw"
        stage_name = "qwen_edit_base"
        lora_path = None
    for index, request, plan in tqdm(prepared, desc="qwen-edit"):
        outputs[index] = run_one_request(
            args,
            config,
            pipe,
            request,
            plan,
            masker,
            final_dir,
            raw_dir,
            masks_dir,
            lora_path,
            stage_name=stage_name,
            candidate_name=candidate_name,
        )
    del pipe
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    outputs = [item for item in outputs if item is not None]
    write_jsonl(output_dir / "predictions.jsonl", outputs)
    print(f"Wrote {len(outputs)} Qwen-Edit-LoRA predictions to {output_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
