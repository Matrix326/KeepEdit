from __future__ import annotations

from pathlib import Path
import os
import subprocess
from typing import Any

import numpy as np
from skimage import filters, morphology

from keepedit.image_ops import diff_mask, full_mask, image_to_float, load_mask, load_rgb, normalize_map, save_mask
from keepedit.io import ensure_dir
from keepedit.schemas import Candidate
from keepedit.schemas import EditPlan, EditRequest


class MaskGenerator:
    def __init__(
        self,
        diff_threshold: float = 0.06,
        min_area_ratio: float = 0.002,
        diff_weight: float = 1.0,
        candidate_variance_weight: float = 0.55,
        qwen_change_weight: float = 0.35,
        close_radius: int = 5,
        dilate_radius: int = 3,
        semantic_command: str | None = None,
        semantic_timeout_seconds: int = 600,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.min_area_ratio = min_area_ratio
        self.diff_weight = diff_weight
        self.candidate_variance_weight = candidate_variance_weight
        self.qwen_change_weight = qwen_change_weight
        self.close_radius = close_radius
        self.dilate_radius = dilate_radius
        self.semantic_command = semantic_command
        self.semantic_timeout_seconds = semantic_timeout_seconds

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MaskGenerator":
        mask_config = config.get("mask", {})
        return cls(
            diff_threshold=float(mask_config.get("diff_threshold", 0.06)),
            min_area_ratio=float(mask_config.get("min_area_ratio", 0.002)),
            diff_weight=float(mask_config.get("diff_weight", 1.0)),
            candidate_variance_weight=float(mask_config.get("candidate_variance_weight", 0.55)),
            qwen_change_weight=float(mask_config.get("qwen_change_weight", 0.35)),
            close_radius=int(mask_config.get("close_radius", 5)),
            dilate_radius=int(mask_config.get("dilate_radius", 3)),
            semantic_command=mask_config.get("semantic_command") or None,
            semantic_timeout_seconds=int(mask_config.get("semantic_timeout_seconds", 600)),
        )

    def generate(
        self,
        request: EditRequest,
        plan: EditPlan,
        out_dir: str | Path,
        reference_image: str | Path | None = None,
        candidates: list[Candidate] | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        out_dir = ensure_dir(out_dir)
        out_path = out_dir / f"{request.id}_mask.png"
        input_image = load_rgb(request.input_image)
        ref_path = reference_image or request.target_image

        if request.mask_image and Path(request.mask_image).exists():
            raw_mask = load_mask(request.mask_image, input_image.size)
            if ref_path and Path(ref_path).exists():
                target_image = load_rgb(ref_path, input_image.size)
                mask, calibration_meta = self._calibrate_dataset_mask(raw_mask, input_image, target_image)
            else:
                mask = self._regularize_mask(raw_mask > 0.5)
                calibration_meta = {
                    "dataset_mask_polarity": "raw_bright_no_reference",
                    "dataset_mask_calibrated": False,
                }
            save_mask(mask, out_path)
            return out_path, {
                "mask_source": "dataset_mask",
                "mask_strategy": "dataset_mask_calibrated_to_edit_region",
                "edit_mask_ratio": float(mask.mean()),
                **calibration_meta,
            }

        if self._is_global_edit(plan, request.instruction):
            mask = full_mask(input_image.size)
            save_mask(mask, out_path)
            return out_path, {
                "mask_source": "full_image_global_edit",
                "mask_strategy": "global_instruction_full_mask",
                "edit_mask_ratio": 1.0,
            }

        if ref_path and Path(ref_path).exists():
            target_image = load_rgb(ref_path, input_image.size)
            mask, pseudo_meta = self._pseudo_mask(input_image, target_image, candidates or [])
            save_mask(mask, out_path)
            return out_path, {
                "mask_source": "target_candidate_difference",
                "mask_strategy": "target_diff_candidate_variance_qwen_change",
                "edit_mask_ratio": float(mask.mean()),
                **pseudo_meta,
            }

        semantic_mask = self._semantic_mask_with_command(request, out_dir, input_image.size)
        if semantic_mask is not None:
            mask, semantic_meta = semantic_mask
            save_mask(mask, out_path)
            return out_path, {
                "mask_source": "semantic_grounding",
                "mask_strategy": "GroundingDINO_SAM_command_hook",
                "edit_mask_ratio": float(mask.mean()),
                **semantic_meta,
            }

        mask = full_mask(input_image.size)
        save_mask(mask, out_path)
        return out_path, {
            "mask_source": "full_image_fallback",
            "mask_strategy": "full_mask_no_reference",
            "edit_mask_ratio": 1.0,
            "reason": f"no mask for {plan.edit_type}",
        }

    def _semantic_mask_with_command(
        self,
        request: EditRequest,
        out_dir: Path,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Run an optional GroundingDINO/SAM style command to produce a mask.

        The command can use either Python format fields or environment variables:

        - `{image}` / `KEEPEDIT_IMAGE`: source image path
        - `{prompt}` / `KEEPEDIT_PROMPT`: edit instruction
        - `{output}` / `KEEPEDIT_OUT_MASK`: expected grayscale mask path

        This keeps semantic segmentation integration real without making heavy
        grounding dependencies mandatory for the MagicBrush target-supervised
        training path.
        """

        if not self.semantic_command:
            return None
        semantic_out = out_dir / f"{request.id}_semantic_grounding_mask.png"
        command = self.semantic_command.format(
            image=str(request.input_image),
            prompt=request.instruction.replace('"', '\\"'),
            output=str(semantic_out),
        )
        env = os.environ.copy()
        env.update(
            {
                "KEEPEDIT_IMAGE": str(request.input_image),
                "KEEPEDIT_PROMPT": request.instruction,
                "KEEPEDIT_OUT_MASK": str(semantic_out),
            }
        )
        try:
            completed = subprocess.run(
                command,
                shell=True,
                env=env,
                cwd=str(Path.cwd()),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.semantic_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None
        if completed.returncode != 0 or not semantic_out.exists():
            return None
        mask = self._regularize_mask(load_mask(semantic_out, image_size) > 0.5)
        if mask.sum() <= 4:
            return None
        return mask, {
            "semantic_command": command,
            "semantic_returncode": completed.returncode,
            "semantic_stdout_tail": completed.stdout[-500:],
            "semantic_stderr_tail": completed.stderr[-500:],
        }

    def _pseudo_mask(
        self,
        input_image: Any,
        target_image: Any,
        candidates: list[Candidate],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        source = image_to_float(input_image)
        target = image_to_float(target_image.resize(input_image.size))
        target_diff = normalize_map(np.abs(target - source).mean(axis=2))

        candidate_arrays = []
        qwen_change = np.zeros(target_diff.shape, dtype=np.float32)
        qwen_used = False
        valid_candidates = []
        for candidate in candidates:
            if candidate.metadata.get("unavailable") or not candidate.image_path.exists():
                continue
            try:
                candidate_array = image_to_float(load_rgb(candidate.image_path, input_image.size))
            except Exception:
                continue
            candidate_arrays.append(candidate_array)
            valid_candidates.append(candidate.name)
            if candidate.name == "qwen_image_edit":
                qwen_change = normalize_map(np.abs(candidate_array - source).mean(axis=2))
                qwen_used = True

        if len(candidate_arrays) >= 2:
            stack = np.stack(candidate_arrays, axis=0)
            candidate_variance = normalize_map(stack.var(axis=0).mean(axis=2))
        elif len(candidate_arrays) == 1:
            candidate_variance = normalize_map(np.abs(candidate_arrays[0] - source).mean(axis=2))
        else:
            candidate_variance = np.zeros(target_diff.shape, dtype=np.float32)

        score = (
            self.diff_weight * target_diff
            + self.candidate_variance_weight * candidate_variance
            + self.qwen_change_weight * qwen_change
        )
        score = normalize_map(score)
        if float(score.max()) <= 1e-8:
            mask = diff_mask(input_image, target_image, threshold=self.diff_threshold, min_area_ratio=self.min_area_ratio)
            return mask, {
                "pseudo_mask_fallback": "target_diff_mask",
                "candidate_mask_experts": valid_candidates,
                "candidate_mask_qwen_used": qwen_used,
            }

        threshold = max(self.diff_threshold, float(filters.threshold_otsu(score)) * 0.85)
        hard = score >= threshold
        min_area = max(8, int(score.size * self.min_area_ratio))
        hard = morphology.remove_small_objects(hard, min_size=min_area)
        hard = morphology.remove_small_holes(hard, area_threshold=min_area)
        if self.close_radius > 0:
            hard = morphology.binary_closing(hard, morphology.disk(self.close_radius))
        if self.dilate_radius > 0:
            hard = morphology.binary_dilation(hard, morphology.disk(self.dilate_radius))
        if hard.sum() == 0:
            cutoff = np.quantile(score, 0.92)
            hard = score >= max(cutoff, self.diff_threshold)
        mask = hard.astype(np.float32)
        return mask, {
            "pseudo_mask_threshold": float(threshold),
            "candidate_mask_experts": valid_candidates,
            "candidate_mask_qwen_used": qwen_used,
            "candidate_mask_mean_score": float(score.mean()),
        }

    @staticmethod
    def _is_global_edit(plan: EditPlan, instruction: str) -> bool:
        if plan.local_or_global == "global" or plan.edit_type in {"style", "background", "global"}:
            return True
        text = instruction.lower()
        global_terms = (
            "style",
            "lighting",
            "light",
            "weather",
            "night",
            "daytime",
            "sunset",
            "snow",
            "rain",
            "fog",
            "color tone",
            "black and white",
            "sepia",
            "cartoon",
            "painting",
            "overall",
            "entire image",
            "whole image",
        )
        return any(term in text for term in global_terms)

    def _regularize_mask(self, mask: np.ndarray) -> np.ndarray:
        hard = mask.astype(bool)
        min_area = max(8, int(hard.size * self.min_area_ratio))
        hard = morphology.remove_small_objects(hard, min_size=min_area)
        hard = morphology.remove_small_holes(hard, area_threshold=min_area)
        if self.close_radius > 0:
            hard = morphology.binary_closing(hard, morphology.disk(self.close_radius))
        if hard.sum() == 0:
            hard = mask.astype(bool)
        return hard.astype(np.float32)

    def _calibrate_dataset_mask(
        self,
        raw_mask: np.ndarray,
        input_image: Any,
        target_image: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Convert arbitrary dataset mask polarity into KeepEdit's 1=edit convention.

        MagicBrush masks are stored as soft grayscale maps whose polarity is not
        consistent with our downstream assumption. We select the polarity and
        threshold whose foreground best overlaps the actual source->target
        difference. This uses target only during training/evaluation teacher
        construction; final deployment does not depend on dataset masks.
        """

        source = image_to_float(input_image)
        target = image_to_float(target_image.resize(input_image.size))
        diff = np.abs(target - source).mean(axis=2)
        diff_gate = max(self.diff_threshold, float(np.quantile(diff, 0.75)))
        diff_hard = diff >= diff_gate
        if diff_hard.sum() == 0:
            diff_hard = diff_mask(input_image, target_image, threshold=self.diff_threshold, min_area_ratio=self.min_area_ratio) > 0.5

        thresholds = {0.2, 0.5, 0.8}
        if float(raw_mask.max() - raw_mask.min()) > 1e-6:
            try:
                thresholds.add(float(filters.threshold_otsu(raw_mask)))
            except Exception:
                pass

        candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
        for threshold in sorted(thresholds):
            for polarity, binary in (
                ("bright_is_edit", raw_mask >= threshold),
                ("dark_is_edit", raw_mask <= threshold),
            ):
                mask = self._regularize_mask(binary)
                score, meta = self._score_mask_against_diff(mask, diff, diff_hard)
                meta.update(
                    {
                        "dataset_mask_polarity": polarity,
                        "dataset_mask_threshold": float(threshold),
                        "dataset_mask_candidate": "raw_dataset_mask",
                    }
                )
                candidates.append((score, mask, meta))

        diff_candidate = self._regularize_mask(diff_hard)
        score, meta = self._score_mask_against_diff(diff_candidate, diff, diff_hard)
        meta.update(
            {
                "dataset_mask_polarity": "target_diff_fallback",
                "dataset_mask_threshold": float(diff_gate),
                "dataset_mask_candidate": "target_diff_mask",
            }
        )
        candidates.append((score, diff_candidate, meta))

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_mask, best_meta = candidates[0]
        best_meta.update(
            {
                "dataset_mask_calibrated": True,
                "dataset_mask_raw_mean": float(raw_mask.mean()),
                "dataset_mask_raw_min": float(raw_mask.min()),
                "dataset_mask_raw_max": float(raw_mask.max()),
                "dataset_mask_calibration_score": float(best_score),
            }
        )
        return best_mask.astype(np.float32), best_meta

    @staticmethod
    def _score_mask_against_diff(mask: np.ndarray, diff: np.ndarray, diff_hard: np.ndarray) -> tuple[float, dict[str, Any]]:
        hard = mask > 0.5
        area = float(hard.mean())
        if hard.sum() == 0 or (~hard).sum() == 0:
            return -1e6, {
                "dataset_mask_area": area,
                "dataset_mask_diff_inside": 0.0,
                "dataset_mask_diff_outside": 0.0,
                "dataset_mask_diff_separation": -1.0,
                "dataset_mask_diff_iou": 0.0,
                "dataset_mask_diff_recall": 0.0,
            }
        inside = float(diff[hard].mean())
        outside = float(diff[~hard].mean())
        separation = inside - outside
        intersection = float((hard & diff_hard).sum())
        union = float((hard | diff_hard).sum())
        target_area = float(diff_hard.sum())
        iou = intersection / max(union, 1.0)
        recall = intersection / max(target_area, 1.0)
        area_penalty = max(0.0, 0.002 - area) * 10.0 + max(0.0, area - 0.96) * 0.25
        score = separation + 0.12 * iou + 0.04 * recall - area_penalty
        return score, {
            "dataset_mask_area": area,
            "dataset_mask_diff_inside": inside,
            "dataset_mask_diff_outside": outside,
            "dataset_mask_diff_separation": separation,
            "dataset_mask_diff_iou": iou,
            "dataset_mask_diff_recall": recall,
        }
