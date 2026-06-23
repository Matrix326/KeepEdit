from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EditRequest:
    id: str
    input_image: Path
    instruction: str
    target_image: Path | None = None
    mask_image: Path | None = None
    edit_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, Any], base_dir: Path | None = None) -> "EditRequest":
        base = base_dir or Path(".")

        def opt_path(key: str) -> Path | None:
            value = item.get(key)
            if not value:
                return None
            path = Path(value)
            if path.is_absolute() or path.exists():
                return path
            return base / path

        input_image = opt_path("input_image") or opt_path("input") or opt_path("source_image")
        if input_image is None:
            raise ValueError(f"Request {item.get('id', '<unknown>')} has no input_image")
        instruction = item.get("instruction") or item.get("edit") or item.get("prompt")
        if not instruction:
            raise ValueError(f"Request {item.get('id', '<unknown>')} has no instruction")

        return cls(
            id=str(item.get("id") or input_image.stem),
            input_image=input_image,
            instruction=str(instruction),
            target_image=opt_path("target_image") or opt_path("edited_image") or opt_path("target"),
            mask_image=opt_path("mask_image") or opt_path("mask"),
            edit_type=item.get("edit_type"),
            metadata=dict(item.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_image": str(self.input_image),
            "instruction": self.instruction,
            "target_image": str(self.target_image) if self.target_image else None,
            "mask_image": str(self.mask_image) if self.mask_image else None,
            "edit_type": self.edit_type,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EditPlan:
    edit_type: str
    target_phrases: list[str] = field(default_factory=list)
    exclude_phrases: list[str] = field(default_factory=list)
    local_or_global: str = "local"
    preservation_level: str = "high"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_type": self.edit_type,
            "target_phrases": self.target_phrases,
            "exclude_phrases": self.exclude_phrases,
            "local_or_global": self.local_or_global,
            "preservation_level": self.preservation_level,
            "notes": self.notes,
        }


@dataclass(slots=True)
class Candidate:
    name: str
    image_path: Path
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, Any], base_dir: Path | None = None) -> "Candidate":
        base = base_dir or Path(".")
        path = Path(item["image_path"])
        if path.is_absolute() or path.exists():
            image_path = path
        else:
            image_path = base / path
        return cls(
            name=str(item["name"]),
            image_path=image_path,
            score=item.get("score"),
            metadata=dict(item.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "image_path": str(self.image_path),
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EditResult:
    id: str
    input_image: Path
    instruction: str
    output_image: Path
    mask_image: Path | None = None
    target_image: Path | None = None
    candidates: list[Candidate] = field(default_factory=list)
    plan: EditPlan | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_image": str(self.input_image),
            "instruction": self.instruction,
            "output_image": str(self.output_image),
            "mask_image": str(self.mask_image) if self.mask_image else None,
            "target_image": str(self.target_image) if self.target_image else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "plan": self.plan.to_dict() if self.plan else None,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
