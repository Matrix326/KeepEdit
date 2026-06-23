from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from skimage import transform

from keepedit.image_ops import image_to_float, load_rgb, mask_boundary_band
from keepedit.schemas import Candidate


EPS = 1e-8


@dataclass(slots=True)
class ExpertScore:
    name: str
    image_path: str
    edit_mse_to_target: float
    edit_ssim_to_target: float
    background_mse_to_source: float
    background_ssim_to_source: float
    background_mse_to_target: float
    full_mse_to_target: float
    input_change_l1: float
    directional_pixel_score: float
    boundary_artifact_score: float
    edit_score: float
    background_score: float
    combined_score: float
    confidence: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(np.float32)
    denom = float(mask.sum() * a.shape[2])
    if denom <= EPS:
        return 0.0
    return float((((a - b) ** 2) * mask[..., None]).sum() / denom)


def masked_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        return float(np.abs(a - b).mean())
    mask = mask.astype(np.float32)
    denom = float(mask.sum() * a.shape[2])
    if denom <= EPS:
        return 0.0
    return float((np.abs(a - b) * mask[..., None]).sum() / denom)


def _mask_bbox(mask: np.ndarray, pad: int = 4) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mask = (mask > 0.5).astype(np.float32)
    if mask.sum() <= 4:
        return 1.0
    bbox = _mask_bbox(mask)
    if bbox is None:
        return 1.0
    ys, xs = bbox
    a_crop = a[ys, xs] * mask[ys, xs, None]
    b_crop = b[ys, xs] * mask[ys, xs, None]
    max_side = max(a_crop.shape[:2])
    if max_side > 192:
        scale = 192 / float(max_side)
        new_h = max(8, int(round(a_crop.shape[0] * scale)))
        new_w = max(8, int(round(a_crop.shape[1] * scale)))
        a_crop = transform.resize(a_crop, (new_h, new_w, 3), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
        b_crop = transform.resize(b_crop, (new_h, new_w, 3), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
    min_side = min(a_crop.shape[:2])
    if min_side < 7:
        mse = float(((a_crop - b_crop) ** 2).mean())
        return float(max(0.0, 1.0 - mse))
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    try:
        return float(
            structural_similarity(
                a_crop,
                b_crop,
                channel_axis=2,
                data_range=1.0,
                win_size=win_size,
            )
        )
    except ValueError:
        mse = float(((a_crop - b_crop) ** 2).mean())
        return float(max(0.0, 1.0 - mse))


def directional_pixel_score(source: np.ndarray, candidate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Fast image-space edit-direction consistency score.

    It compares the image-space edit direction `(candidate - source)` with
    `(target - source)` inside the edit mask. The value is mapped to [0, 1].
    """

    mask3 = (mask > 0.5).astype(np.float32)[..., None]
    cand_delta = ((candidate - source) * mask3).reshape(-1)
    target_delta = ((target - source) * mask3).reshape(-1)
    cand_norm = float(np.linalg.norm(cand_delta))
    target_norm = float(np.linalg.norm(target_delta))
    if cand_norm <= EPS or target_norm <= EPS:
        return 0.0
    cosine = float(np.dot(cand_delta, target_delta) / (cand_norm * target_norm + EPS))
    return float(np.clip((cosine + 1.0) * 0.5, 0.0, 1.0))


def boundary_artifact_score(source: np.ndarray, candidate: np.ndarray, mask: np.ndarray, radius: int = 6) -> float:
    """Lower is better. Penalize color and gradient jumps around the edit boundary."""

    band = mask_boundary_band(mask, radius=radius)
    if band.sum() <= 4:
        return 0.0
    color_jump = masked_l1(candidate, source, band)
    source_gray = source.mean(axis=2)
    candidate_gray = candidate.mean(axis=2)
    sy, sx = np.gradient(source_gray)
    cy, cx = np.gradient(candidate_gray)
    grad_source = np.sqrt(sx * sx + sy * sy)
    grad_candidate = np.sqrt(cx * cx + cy * cy)
    grad_jump = masked_l1(grad_candidate[..., None], grad_source[..., None], band)
    return float(0.65 * color_jump + 0.35 * grad_jump)


def _relative_confidences(scores: list[float]) -> list[float]:
    if not scores:
        return []
    order = np.argsort(np.asarray(scores, dtype=np.float32))
    confidences = [0.5 for _ in scores]
    if len(order) == 1:
        confidences[int(order[0])] = 0.6
        return confidences
    best_idx = int(order[0])
    second_idx = int(order[1])
    best = float(scores[best_idx])
    second = float(scores[second_idx])
    scale = max(abs(best), abs(second), 1e-4)
    gap = max(0.0, second - best)
    confidences[best_idx] = float(np.clip(0.55 + 0.45 * np.tanh(gap / (0.25 * scale + EPS)), 0.55, 1.0))
    return confidences


def score_candidate_arrays(
    source: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    edit_mask: np.ndarray,
    name: str,
    image_path: str | Path,
    edit_weight: float = 1.0,
    background_weight: float = 0.35,
    semantic_weight: float = 0.25,
    boundary_weight: float = 0.20,
) -> ExpertScore:
    edit_mask = (edit_mask > 0.5).astype(np.float32)
    bg_mask = 1.0 - edit_mask

    edit_mse = masked_mse(candidate, target, edit_mask)
    edit_ssim = masked_ssim(candidate, target, edit_mask)
    bg_mse_source = masked_mse(candidate, source, bg_mask) if bg_mask.sum() > 0 else 0.0
    bg_ssim_source = masked_ssim(candidate, source, bg_mask) if bg_mask.sum() > 0 else 1.0
    bg_mse_target = masked_mse(candidate, target, bg_mask) if bg_mask.sum() > 0 else 0.0
    full_mse = float(((candidate - target) ** 2).mean())
    change_l1 = masked_l1(candidate, source)
    direction = directional_pixel_score(source, candidate, target, edit_mask)
    boundary = boundary_artifact_score(source, candidate, edit_mask)

    edit_score = edit_mse + 0.20 * (1.0 - edit_ssim) + semantic_weight * (1.0 - direction)
    background_score = bg_mse_source + 0.20 * (1.0 - bg_ssim_source)
    combined_score = edit_weight * edit_score + background_weight * background_score + boundary_weight * boundary + 0.10 * full_mse
    return ExpertScore(
        name=name,
        image_path=str(image_path),
        edit_mse_to_target=float(edit_mse),
        edit_ssim_to_target=float(edit_ssim),
        background_mse_to_source=float(bg_mse_source),
        background_ssim_to_source=float(bg_ssim_source),
        background_mse_to_target=float(bg_mse_target),
        full_mse_to_target=float(full_mse),
        input_change_l1=float(change_l1),
        directional_pixel_score=float(direction),
        boundary_artifact_score=float(boundary),
        edit_score=float(edit_score),
        background_score=float(background_score),
        combined_score=float(combined_score),
        confidence=0.5,
    )


def score_experts(
    input_image: str | Path,
    target_image: str | Path,
    candidates: list[Candidate],
    edit_mask: np.ndarray,
    edit_weight: float = 1.0,
    background_weight: float = 0.35,
    semantic_weight: float = 0.25,
    boundary_weight: float = 0.20,
    virtual_qwen_family: bool = False,
    qwen_conservative_strength: float = 0.65,
    qwen_aggressive_strength: float = 1.25,
) -> tuple[list[ExpertScore], dict[str, np.ndarray]]:
    source_image = load_rgb(input_image)
    target_pil = load_rgb(target_image, source_image.size)
    source = image_to_float(source_image)
    target = image_to_float(target_pil)
    arrays: dict[str, np.ndarray] = {}
    scores: list[ExpertScore] = []
    for candidate in candidates:
        if candidate.metadata.get("unavailable") or not candidate.image_path.exists():
            continue
        image = load_rgb(candidate.image_path, source_image.size)
        array = image_to_float(image)
        arrays[candidate.name] = array
        scores.append(
            score_candidate_arrays(
                source=source,
                candidate=array,
                target=target,
                edit_mask=edit_mask,
                name=candidate.name,
                image_path=candidate.image_path,
                edit_weight=edit_weight,
                background_weight=background_weight,
                semantic_weight=semantic_weight,
                boundary_weight=boundary_weight,
            )
        )
        if virtual_qwen_family and candidate.name == "qwen_image_edit":
            variants = {
                "qwen_image_edit_conservative": np.clip(source + qwen_conservative_strength * (array - source), 0.0, 1.0),
                "qwen_image_edit_aggressive": np.clip(source + qwen_aggressive_strength * (array - source), 0.0, 1.0),
            }
            for variant_name, variant_array in variants.items():
                arrays[variant_name] = variant_array
                scores.append(
                    score_candidate_arrays(
                        source=source,
                        candidate=variant_array,
                        target=target,
                        edit_mask=edit_mask,
                        name=variant_name,
                        image_path=candidate.image_path,
                        edit_weight=edit_weight,
                        background_weight=background_weight,
                        semantic_weight=semantic_weight,
                        boundary_weight=boundary_weight,
                    )
                )

    confidences = _relative_confidences([item.combined_score for item in scores])
    for item, confidence in zip(scores, confidences):
        item.confidence = confidence
    return scores, arrays
