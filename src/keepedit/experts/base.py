from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from keepedit.io import ensure_dir
from keepedit.schemas import Candidate, EditRequest


class BaseExpert(ABC):
    name: str

    def __init__(self, name: str, **_: Any) -> None:
        self.name = name

    @abstractmethod
    def generate(self, request: EditRequest, out_dir: str | Path) -> Candidate:
        raise NotImplementedError


class IdentityExpert(BaseExpert):
    def generate(self, request: EditRequest, out_dir: str | Path) -> Candidate:
        out_dir = ensure_dir(out_dir)
        suffix = request.input_image.suffix or ".png"
        out_path = out_dir / f"{request.id}_{self.name}{suffix}"
        shutil.copyfile(request.input_image, out_path)
        return Candidate(name=self.name, image_path=out_path, metadata={"expert": "identity"})


def build_expert(name: str, config: dict[str, Any]) -> BaseExpert:
    expert_type = config.get("type", name)
    if expert_type == "identity":
        return IdentityExpert(name=name)
    if expert_type == "instruct_pix2pix":
        from keepedit.experts.pix2pix_diffusers import DiffusersInstructPix2PixExpert

        return DiffusersInstructPix2PixExpert(name=name, **config)
    if expert_type == "qwen_image_edit":
        from keepedit.experts.qwen_image_edit import QwenImageEditExpert

        return QwenImageEditExpert(name=name, **config)
    if expert_type == "editar":
        from keepedit.experts.editar_native import EditARExpert

        return EditARExpert(name=name, **config)
    raise ValueError(f"Unknown expert type: {expert_type}")
