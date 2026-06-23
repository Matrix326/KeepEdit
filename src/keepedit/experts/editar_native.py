from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from keepedit.experts.base import BaseExpert
from keepedit.io import ensure_dir
from keepedit.schemas import Candidate, EditRequest


class EditARExpert(BaseExpert):
    """Native wrapper around the official JitengMu/EditAR inference code."""

    def __init__(
        self,
        name: str,
        repo_dir: str = "external/EditAR",
        gpt_ckpt: str = "external/EditAR/checkpoints/editar/editar_release.pt",
        vq_ckpt: str = "external/EditAR/pretrained_models/vq_ds16_t2i.pt",
        t5_path: str = "external/EditAR/pretrained_models/t5-ckpt",
        t5_model_type: str = "flan-t5-xl",
        cfg_scale: float = 3.0,
        seed: int = 83,
        image_size: int = 512,
        mixed_precision: str = "bf16",
        optional: bool = False,
        **_: Any,
    ) -> None:
        super().__init__(name)
        self.repo_dir = Path(repo_dir)
        self.gpt_ckpt = Path(gpt_ckpt)
        self.vq_ckpt = Path(vq_ckpt)
        self.t5_path = Path(t5_path)
        self.t5_model_type = t5_model_type
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.image_size = image_size
        self.mixed_precision = mixed_precision
        self.optional = optional
        self._loaded: dict[str, Any] | None = None

    def _missing_reason(self) -> str | None:
        required = {
            "repo_dir": self.repo_dir,
            "gpt_ckpt": self.gpt_ckpt,
            "vq_ckpt": self.vq_ckpt,
            "t5_model": self.t5_path / self.t5_model_type,
        }
        missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
        return "; ".join(missing) if missing else None

    def _unavailable(self, request: EditRequest, out_path: Path, reason: str) -> Candidate:
        Image.open(request.input_image).convert("RGB").save(out_path)
        return Candidate(
            name=self.name,
            image_path=out_path,
            metadata={"expert": "editar", "unavailable": True, "reason": reason},
        )

    @staticmethod
    def _vqgan_input_from(img: Image.Image, target_image_size: int):
        import torch

        img = img.convert("RGB")
        side = min(img.size)
        scale = target_image_size / side
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))
        array = np.asarray(img, dtype=np.float32) / 255.0
        array = array * 2.0 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).float()

    @staticmethod
    def _tensor_to_image(sample: Any) -> Image.Image:
        sample = sample.detach().float().clamp(-1, 1)
        array = ((sample + 1.0) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array, mode="RGB")

    def _load(self) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded

        missing = self._missing_reason()
        if missing:
            raise FileNotFoundError(f"EditAR assets are incomplete: {missing}")

        import torch

        repo = str(self.repo_dir.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

        # EditAR disables default initialization in its sample script to avoid
        # spending minutes initializing tensors that will immediately be loaded.
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

        from autoregressive.models.generate_edit import generate
        from autoregressive.models.gpt_edit import GPT_models
        from language.t5 import T5Embedder
        from tokenizer.tokenizer_image.vq_model import VQ_models

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(self.seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[self.mixed_precision]
        latent_size = self.image_size // 16

        vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8).to(device)
        vq_model.eval()
        checkpoint = torch.load(self.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        del checkpoint

        gpt_model = GPT_models["GPT-XL"](
            vocab_size=16384,
            block_size=latent_size**2,
            num_classes=1000,
            cls_token_num=120,
            model_type="edit",
            model_mode="joint_cls_emb",
            resid_dropout_p=0.1,
            ffn_dropout_p=0.1,
            token_dropout_p=0.1,
            distill_mode=None,
        ).to(device=device, dtype=precision)
        checkpoint = torch.load(self.gpt_ckpt, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "module" in checkpoint:
            state_dict = checkpoint["module"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        gpt_model.load_state_dict(state_dict, strict=False)
        gpt_model.eval()
        del checkpoint

        t5_model = T5Embedder(
            device=device,
            local_cache=True,
            cache_dir=str(self.t5_path),
            dir_or_name=self.t5_model_type,
            torch_dtype=precision,
            model_max_length=120,
        )

        self._loaded = {
            "torch": torch,
            "device": device,
            "generate": generate,
            "vq_model": vq_model,
            "gpt_model": gpt_model,
            "t5_model": t5_model,
            "latent_size": latent_size,
            "codebook_embed_dim": 8,
        }
        return self._loaded

    def generate(self, request: EditRequest, out_dir: str | Path) -> Candidate:
        out_dir = ensure_dir(out_dir)
        out_path = out_dir / f"{request.id}_{self.name}.png"
        missing = self._missing_reason()
        if missing and self.optional:
            return self._unavailable(request, out_path, missing)

        t0 = time.time()
        state = self._load()
        torch = state["torch"]
        device = state["device"]

        input_image = Image.open(request.input_image).convert("RGB")
        input_tensor = self._vqgan_input_from(input_image, self.image_size)[None].to(device, non_blocking=True)
        tokens = state["t5_model"].tokenizer(
            request.instruction,
            max_length=120,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attn_mask = tokens["attention_mask"].to(device)
        input_mode = torch.ones((1,), dtype=torch.long, device=device)

        with torch.no_grad():
            text_embs = state["t5_model"].model(input_ids=input_ids, attention_mask=attn_mask)["last_hidden_state"].detach()
            _, _, [_, _, input_indices] = state["vq_model"].encode(input_tensor)
            input_indices = input_indices.reshape(input_tensor.shape[0], -1)
            index_sample = state["generate"](
                state["gpt_model"],
                text_embs,
                input_indices,
                input_mode,
                state["latent_size"] ** 2,
                emb_masks=None,
                cfg_scale=self.cfg_scale,
                temperature=1.0,
                top_k=1000,
                top_p=1.0,
                sample_logits=True,
            )
            qzshape = [1, state["codebook_embed_dim"], state["latent_size"], state["latent_size"]]
            sample = state["vq_model"].decode_code(index_sample, qzshape)[0]

        image = self._tensor_to_image(sample)
        image.save(out_path)
        return Candidate(
            name=self.name,
            image_path=out_path,
            metadata={
                "expert": "editar",
                "cfg_scale": self.cfg_scale,
                "seed": self.seed,
                "seconds": round(time.time() - t0, 3),
            },
        )
