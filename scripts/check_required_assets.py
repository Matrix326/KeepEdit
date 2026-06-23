#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED = {
    "pix2pix_model_index": (Path("checkpoints/hf/timbrooks__instruct-pix2pix/model_index.json"), 1),
    "pix2pix_unet": (
        Path("checkpoints/hf/timbrooks__instruct-pix2pix/unet/diffusion_pytorch_model.fp16.safetensors"),
        1_000_000_000,
    ),
    "pix2pix_vae": (
        Path("checkpoints/hf/timbrooks__instruct-pix2pix/vae/diffusion_pytorch_model.fp16.safetensors"),
        100_000_000,
    ),
    "pix2pix_text_encoder": (
        Path("checkpoints/hf/timbrooks__instruct-pix2pix/text_encoder/model.fp16.safetensors"),
        100_000_000,
    ),
    "stage1_qwen_edit_index": (
        Path("checkpoints/hf/Qwen__Qwen-Image-Edit/transformer/diffusion_pytorch_model.safetensors.index.json"),
        1,
    ),
    "stage1_qwen_edit_transformer_1": (
        Path("checkpoints/hf/Qwen__Qwen-Image-Edit/transformer/diffusion_pytorch_model-00001-of-00009.safetensors"),
        1_000_000_000,
    ),
    "stage1_qwen_edit_transformer_9": (
        Path("checkpoints/hf/Qwen__Qwen-Image-Edit/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"),
        1_000_000_000,
    ),
    "stage1_qwen_edit_text": (
        Path("checkpoints/hf/Qwen__Qwen-Image-Edit/text_encoder/model-00001-of-00004.safetensors"),
        1_000_000_000,
    ),
    "stage1_qwen_edit_vae": (
        Path("checkpoints/hf/Qwen__Qwen-Image-Edit/vae/diffusion_pytorch_model.safetensors"),
        100_000_000,
    ),
    "editar_release": (Path("external/EditAR/checkpoints/editar/editar_release.pt"), 9_385_899_080),
    "editar_vq": (Path("external/EditAR/pretrained_models/vq_ds16_t2i.pt"), 280_000_000),
    "editar_t2i": (Path("external/EditAR/pretrained_models/t2i_XL_stage2_512.pt"), 2_000_000_000),
    "editar_t5_config": (Path("external/EditAR/pretrained_models/t5-ckpt/flan-t5-xl/config.json"), 1),
    "editar_t5_index": (Path("external/EditAR/pretrained_models/t5-ckpt/flan-t5-xl/model.safetensors.index.json"), 1),
    "editar_t5_1": (
        Path("external/EditAR/pretrained_models/t5-ckpt/flan-t5-xl/model-00001-of-00002.safetensors"),
        1_000_000_000,
    ),
    "editar_t5_2": (
        Path("external/EditAR/pretrained_models/t5-ckpt/flan-t5-xl/model-00002-of-00002.safetensors"),
        1_000_000_000,
    ),
    "qwen_edit_transformer": (
        Path("checkpoints/diffsynth/Qwen/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00001-of-00005.safetensors"),
        4_000_000_000,
    ),
    "qwen_edit_transformer_2": (
        Path("checkpoints/diffsynth/Qwen/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00002-of-00005.safetensors"),
        4_000_000_000,
    ),
    "qwen_edit_transformer_3": (
        Path("checkpoints/diffsynth/Qwen/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00003-of-00005.safetensors"),
        4_000_000_000,
    ),
    "qwen_edit_transformer_4": (
        Path("checkpoints/diffsynth/Qwen/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00004-of-00005.safetensors"),
        4_000_000_000,
    ),
    "qwen_edit_transformer_5": (
        Path("checkpoints/diffsynth/Qwen/Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model-00005-of-00005.safetensors"),
        900_000_000,
    ),
    "qwen_image_text": (
        Path("checkpoints/diffsynth/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors"),
        4_000_000_000,
    ),
    "qwen_image_text_2": (
        Path("checkpoints/diffsynth/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors"),
        4_000_000_000,
    ),
    "qwen_image_text_3": (
        Path("checkpoints/diffsynth/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors"),
        4_000_000_000,
    ),
    "qwen_image_text_4": (
        Path("checkpoints/diffsynth/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"),
        1_000_000_000,
    ),
    "qwen_image_vae": (
        Path("checkpoints/diffsynth/Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors"),
        100_000_000,
    ),
    "qwen3_vl_config": (Path("checkpoints/hf/Qwen3-VL-8B-Instruct/config.json"), 1),
    "qwen3_vl_index": (Path("checkpoints/hf/Qwen3-VL-8B-Instruct/model.safetensors.index.json"), 1),
    "qwen3_vl_1": (Path("checkpoints/hf/Qwen3-VL-8B-Instruct/model-00001-of-00004.safetensors"), 1_000_000_000),
    "qwen3_vl_2": (Path("checkpoints/hf/Qwen3-VL-8B-Instruct/model-00002-of-00004.safetensors"), 1_000_000_000),
    "qwen3_vl_3": (Path("checkpoints/hf/Qwen3-VL-8B-Instruct/model-00003-of-00004.safetensors"), 1_000_000_000),
    "qwen3_vl_4": (Path("checkpoints/hf/Qwen3-VL-8B-Instruct/model-00004-of-00004.safetensors"), 1_000_000_000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether required KeepEdit assets are present.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def ok(path: Path, min_size: int) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if Path(str(path) + ".aria2").exists():
        return False, "aria2_incomplete"
    size = path.stat().st_size
    if size < min_size:
        return False, f"too_small={size}"
    return True, f"size={size}"


def main() -> None:
    args = parse_args()
    all_ok = True
    for name, (path, min_size) in REQUIRED.items():
        ready, reason = ok(path, min_size)
        all_ok = all_ok and ready
        if not args.quiet:
            print(f"{'OK' if ready else 'WAIT'} {name}: {path} {reason}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
