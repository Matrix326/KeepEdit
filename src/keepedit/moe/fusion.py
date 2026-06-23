from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage import measure, morphology

from keepedit.image_ops import (
    dilate_mask,
    feather_mask,
    float_to_image,
    image_to_float,
    load_mask,
    load_rgb,
    save_image,
    save_mask,
    seamless_blend,
)
from keepedit.io import ensure_dir
from keepedit.moe.local_refiner import LocalRefiner, LocalRefinerConfig
from keepedit.moe.scoring import ExpertScore, masked_mse, masked_ssim, score_candidate_arrays, score_experts
from keepedit.schemas import Candidate, EditPlan, EditRequest


PALETTE = {
    "source": (238, 238, 238),
    "pix2pix": (0, 145, 255),
    "qwen_image_edit": (255, 99, 71),
    "qwen_image_edit_conservative": (255, 178, 102),
    "qwen_image_edit_aggressive": (220, 20, 60),
    "editar": (126, 87, 194),
    "identity": (130, 130, 130),
    "unknown": (38, 166, 154),
}


@dataclass(slots=True)
class RankCalibrator:
    enabled: bool = False
    expert_bias: dict[str, float] = field(default_factory=dict)
    edit_mse_weight: float = 0.0
    bg_mse_weight: float = 0.0
    direction_weight: float = 0.0
    boundary_weight: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RankCalibrator":
        moe = config.get("stage1_moe_fusion", {})
        raw = moe.get("rank_calibrator") or {}
        if not raw:
            return cls()
        loaded: dict[str, Any] = {}
        path = raw.get("path")
        if path:
            try:
                import json

                loaded = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                loaded = {}
        merged = {**raw, **loaded}
        return cls(
            enabled=bool(merged.get("enabled", True)),
            expert_bias={str(k): float(v) for k, v in (merged.get("expert_bias") or {}).items()},
            edit_mse_weight=float(merged.get("edit_mse_weight", 0.0)),
            bg_mse_weight=float(merged.get("bg_mse_weight", 0.0)),
            direction_weight=float(merged.get("direction_weight", 0.0)),
            boundary_weight=float(merged.get("boundary_weight", 0.0)),
        )

    def apply(self, raw_score: float, base_score: ExpertScore) -> float:
        if not self.enabled:
            return raw_score
        bias = self.expert_bias.get(base_score.name, 0.0)
        if base_score.name.startswith("qwen_image_edit"):
            bias += self.expert_bias.get("qwen_image_edit*", 0.0)
        calibrated = (
            raw_score
            + bias
            + self.edit_mse_weight * base_score.edit_mse_to_target
            + self.bg_mse_weight * base_score.background_mse_to_source
            + self.direction_weight * (1.0 - base_score.directional_pixel_score)
            + self.boundary_weight * base_score.boundary_artifact_score
        )
        return float(calibrated)


@dataclass(slots=True)
class MoEFusionConfig:
    dilate_radius: int = 8
    feather_radius: int = 12
    component_min_area_ratio: float = 0.001
    edit_weight: float = 1.0
    background_weight: float = 0.35
    semantic_weight: float = 0.25
    boundary_weight: float = 0.20
    color_harmonize_strength: float = 0.25
    global_mask_ratio: float = 0.65
    source_background_for_local: bool = True
    qwen_prior_bonus: float = 0.015
    temperature: float = 0.02
    soft_top_k: int = 1
    pyramid_levels: int = 5
    poisson_blend: bool = False
    poisson_mode: str = "mixed"
    low_confidence_target_mix: float = 0.0
    low_confidence_threshold: float = 0.62
    hard_route_confidence: float = 0.82
    preference_margin: float = 0.01
    fusion_fallback_epsilon: float = 0.005
    fusion_fallback: str = "best_expert"
    virtual_qwen_family: bool = True
    qwen_conservative_strength: float = 0.65
    qwen_aggressive_strength: float = 1.25
    min_edit_change_ratio: float = 0.30
    min_target_edit_change: float = 0.05
    low_edit_fallback: str = "canonical_target"
    local_refiner: LocalRefinerConfig = field(default_factory=LocalRefinerConfig)
    rank_calibrator: RankCalibrator = field(default_factory=RankCalibrator)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MoEFusionConfig":
        composer = config.get("composer", {})
        moe = config.get("stage1_moe_fusion", {})
        return cls(
            dilate_radius=int(moe.get("dilate_radius", composer.get("dilate_radius", 8))),
            feather_radius=int(moe.get("feather_radius", composer.get("feather_radius", 12))),
            component_min_area_ratio=float(moe.get("component_min_area_ratio", 0.001)),
            edit_weight=float(moe.get("edit_weight", 1.0)),
            background_weight=float(moe.get("background_weight", 0.35)),
            semantic_weight=float(moe.get("semantic_weight", 0.25)),
            boundary_weight=float(moe.get("boundary_weight", 0.20)),
            color_harmonize_strength=float(moe.get("color_harmonize_strength", 0.25)),
            global_mask_ratio=float(moe.get("global_mask_ratio", 0.65)),
            source_background_for_local=bool(moe.get("source_background_for_local", True)),
            qwen_prior_bonus=float(moe.get("qwen_prior_bonus", 0.015)),
            temperature=float(moe.get("temperature", 0.02)),
            soft_top_k=int(moe.get("soft_top_k", 1)),
            pyramid_levels=int(moe.get("pyramid_levels", 5)),
            poisson_blend=bool(moe.get("poisson_blend", False)),
            poisson_mode=str(moe.get("poisson_mode", "mixed")),
            low_confidence_target_mix=float(moe.get("low_confidence_target_mix", 0.0)),
            low_confidence_threshold=float(moe.get("low_confidence_threshold", 0.62)),
            hard_route_confidence=float(moe.get("hard_route_confidence", 0.82)),
            preference_margin=float(moe.get("preference_margin", 0.01)),
            fusion_fallback_epsilon=float(moe.get("fusion_fallback_epsilon", 0.005)),
            fusion_fallback=str(moe.get("fusion_fallback", "best_expert")),
            virtual_qwen_family=bool(moe.get("virtual_qwen_family", True)),
            qwen_conservative_strength=float(moe.get("qwen_conservative_strength", 0.65)),
            qwen_aggressive_strength=float(moe.get("qwen_aggressive_strength", 1.25)),
            min_edit_change_ratio=float(moe.get("min_edit_change_ratio", 0.30)),
            min_target_edit_change=float(moe.get("min_target_edit_change", 0.05)),
            low_edit_fallback=str(moe.get("low_edit_fallback", "canonical_target")),
            local_refiner=LocalRefinerConfig.from_config(config),
            rank_calibrator=RankCalibrator.from_config(config),
        )


@dataclass(slots=True)
class MoEFusionResult:
    image: Image.Image
    attribution_map: Image.Image
    confidence_map: Image.Image
    scores: list[ExpertScore]
    selected_bg_expert: str
    selected_edit_experts: list[dict[str, Any]] = field(default_factory=list)
    teacher_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _expert_color(name: str) -> tuple[int, int, int]:
    if name.startswith("qwen_image_edit") and name not in PALETTE:
        return PALETTE["qwen_image_edit"]
    return PALETTE.get(name, PALETTE["unknown"])


def _valid_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    labels = measure.label(mask > 0.5, connectivity=2)
    components = []
    for region in measure.regionprops(labels):
        if region.area < min_area:
            continue
        components.append((labels == region.label).astype(np.float32))
    if not components and mask.sum() > 0:
        components.append((mask > 0.5).astype(np.float32))
    return components


def _component_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _component_score(
    source: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    component_mask: np.ndarray,
    base_score: ExpertScore,
    semantic_weight: float,
    boundary_weight: float,
    qwen_prior_bonus: float,
    rank_calibrator: RankCalibrator | None = None,
) -> float:
    edit_mse = masked_mse(candidate, target, component_mask)
    edit_ssim = masked_ssim(candidate, target, component_mask)
    source_change = masked_mse(candidate, source, component_mask)
    # Encourage actual edits while keeping the target-supervised term dominant.
    no_change_penalty = max(0.0, 0.01 - source_change) * 2.0
    raw_score = float(
        edit_mse
        + 0.15 * (1.0 - edit_ssim)
        + semantic_weight * 0.30 * (1.0 - base_score.directional_pixel_score)
        + boundary_weight * base_score.boundary_artifact_score
        + no_change_penalty
        - (qwen_prior_bonus if base_score.name.startswith("qwen_image_edit") else 0.0)
    )
    if rank_calibrator is not None:
        return rank_calibrator.apply(raw_score, base_score)
    return raw_score


def _region_preference_pairs(
    request_id: str,
    component_id: int,
    component_scores: list[tuple[float, str]],
    margin: float,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for winner_index, (winner_score, winner_name) in enumerate(component_scores):
        for loser_score, loser_name in component_scores[winner_index + 1 :]:
            score_margin = float(loser_score - winner_score)
            if score_margin < margin:
                continue
            pairs.append(
                {
                    "id": f"{request_id}__region{component_id}__{winner_name}_gt_{loser_name}",
                    "component_id": component_id,
                    "winner": winner_name,
                    "loser": loser_name,
                    "winner_reward": float(-winner_score),
                    "loser_reward": float(-loser_score),
                    "reward_margin": score_margin,
                    "pair_type": "region_expert_preference",
                }
            )
    return pairs


def _score_confidence(best: float, second: float | None) -> float:
    if second is None:
        return 0.60
    gap = max(0.0, second - best)
    scale = max(abs(best), abs(second), 1e-4)
    return float(np.clip(0.55 + 0.45 * np.tanh(gap / (0.25 * scale + 1e-8)), 0.55, 1.0))


def _soft_weights(ranked_scores: list[tuple[float, str]], temperature: float, top_k: int) -> dict[str, float]:
    selected = ranked_scores[: max(1, min(top_k, len(ranked_scores)))]
    if len(selected) == 1:
        return {selected[0][1]: 1.0}
    values = np.asarray([item[0] for item in selected], dtype=np.float32)
    values = values - float(values.min())
    logits = -values / max(float(temperature), 1e-6)
    logits = logits - float(logits.max())
    probs = np.exp(logits)
    probs = probs / max(float(probs.sum()), 1e-8)
    return {name: float(weight) for weight, (_, name) in zip(probs, selected)}


def _masked_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    hard = mask > 0.5
    if hard.sum() <= 4:
        return float(np.abs(a - b).mean())
    return float(np.abs(a[hard] - b[hard]).mean())


def _choose_background(
    source: np.ndarray,
    target: np.ndarray,
    arrays: dict[str, np.ndarray],
    scores_by_name: dict[str, ExpertScore],
    bg_mask: np.ndarray,
    global_like: bool,
) -> tuple[str, np.ndarray, float]:
    if bg_mask.sum() <= 4:
        best = min(scores_by_name.values(), key=lambda item: item.full_mse_to_target)
        return best.name, arrays[best.name], best.confidence

    if global_like:
        source_value = masked_mse(source, target, bg_mask)
    else:
        source_value = 0.0
    candidates: list[tuple[str, np.ndarray, float]] = [("source", source, float(source_value))]
    for name, array in arrays.items():
        score = scores_by_name[name]
        if global_like:
            value = 0.70 * score.background_mse_to_target + 0.30 * score.background_mse_to_source
        else:
            value = score.background_score
        candidates.append((name, array, float(value)))
    candidates.sort(key=lambda item: item[2])
    best_name, best_array, best_score = candidates[0]
    second = candidates[1][2] if len(candidates) > 1 else None
    return best_name, best_array, _score_confidence(best_score, second)


def _harmonize(candidate: np.ndarray, background: np.ndarray, component_mask: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return candidate
    hard = component_mask > 0.5
    if hard.sum() <= 4:
        return candidate
    boundary = morphology.binary_dilation(hard, morphology.disk(6)) & ~hard
    if boundary.sum() <= 4:
        return candidate
    delta = background[boundary].mean(axis=0) - candidate[boundary].mean(axis=0)
    return np.clip(candidate + delta.reshape(1, 1, 3) * strength, 0.0, 1.0)


def _save_attribution(path: Path, attribution: np.ndarray) -> Path:
    ensure_dir(path.parent)
    Image.fromarray(np.clip(attribution, 0, 255).astype(np.uint8), mode="RGB").save(path)
    return path


def _save_confidence(path: Path, confidence: np.ndarray) -> Path:
    ensure_dir(path.parent)
    Image.fromarray(np.clip(confidence * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)
    return path


def _persist_virtual_experts(
    request_id: str,
    output_image_path: str | Path,
    arrays: dict[str, np.ndarray],
    scores: list[ExpertScore],
) -> None:
    """Materialize generated expert variants so downstream preference data uses real image paths."""

    out_dir = ensure_dir(Path(output_image_path).parent / "virtual_experts")
    for score in scores:
        if not score.name.startswith("qwen_image_edit_"):
            continue
        array = arrays.get(score.name)
        if array is None:
            continue
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in score.name)
        path = out_dir / f"{request_id}_{safe_name}.png"
        save_image(float_to_image(array), path)
        score.image_path = str(path.resolve())


def build_moe_fusion_teacher(
    request: EditRequest,
    candidates: list[Candidate],
    plan: EditPlan,
    mask_path: str | Path,
    output_image_path: str | Path,
    attribution_path: str | Path,
    confidence_path: str | Path,
    config: MoEFusionConfig,
    canonical_target_path: str | Path | None = None,
) -> MoEFusionResult:
    if request.target_image is None:
        raise ValueError(f"MoE-Fusion teacher requires target_image for request {request.id}")

    source_pil = load_rgb(request.input_image)
    target_pil = load_rgb(request.target_image, source_pil.size)
    source = image_to_float(source_pil)
    target = image_to_float(target_pil)
    edit_mask = (load_mask(mask_path, source_pil.size) > 0.5).astype(np.float32)
    edit_ratio = float(edit_mask.mean())
    global_like = plan.local_or_global == "global" or plan.edit_type in {"style", "background", "global"} or edit_ratio >= config.global_mask_ratio

    scores, arrays = score_experts(
        request.input_image,
        request.target_image,
        candidates,
        edit_mask,
        edit_weight=config.edit_weight,
        background_weight=config.background_weight,
        semantic_weight=config.semantic_weight,
        boundary_weight=config.boundary_weight,
        virtual_qwen_family=config.virtual_qwen_family,
        qwen_conservative_strength=config.qwen_conservative_strength,
        qwen_aggressive_strength=config.qwen_aggressive_strength,
    )
    if not scores:
        raise RuntimeError(f"No valid expert candidates for MoE fusion request {request.id}")
    _persist_virtual_experts(request.id, output_image_path, arrays, scores)
    scores_by_name = {item.name: item for item in scores}
    bg_mask = 1.0 - edit_mask
    if config.source_background_for_local and not global_like:
        selected_bg, bg_array, bg_confidence = "source", source, 1.0
    else:
        selected_bg, bg_array, bg_confidence = _choose_background(
            source,
            target,
            arrays,
            scores_by_name,
            bg_mask,
            global_like=global_like,
        )

    foreground = bg_array.copy()
    attribution = np.zeros((*edit_mask.shape, 3), dtype=np.uint8)
    attribution[:, :] = _expert_color(selected_bg)
    confidence = np.ones(edit_mask.shape, dtype=np.float32) * bg_confidence

    min_area = max(8, int(edit_mask.size * config.component_min_area_ratio))
    components = _valid_components(edit_mask, min_area=min_area)
    selected_components: list[dict[str, Any]] = []
    region_preference_pairs: list[dict[str, Any]] = []

    for component_id, component in enumerate(components):
        component_scores = []
        for score in scores:
            value = _component_score(
                source=source,
                candidate=arrays[score.name],
                target=target,
                component_mask=component,
                base_score=score,
                semantic_weight=config.semantic_weight,
                boundary_weight=config.boundary_weight,
                qwen_prior_bonus=config.qwen_prior_bonus,
                rank_calibrator=config.rank_calibrator,
            )
            component_scores.append((value, score.name))
        component_scores.sort(key=lambda item: item[0])
        best_score, best_name = component_scores[0]
        second_score = component_scores[1][0] if len(component_scores) > 1 else None
        component_confidence = _score_confidence(best_score, second_score)
        if component_confidence >= config.hard_route_confidence:
            soft_weights = {best_name: 1.0}
        else:
            soft_weights = _soft_weights(component_scores, config.temperature, config.soft_top_k)
        candidate_array = np.zeros_like(source)
        for name, weight in soft_weights.items():
            candidate_array += arrays[name] * weight
        if (
            config.low_confidence_target_mix > 0
            and component_confidence < config.low_confidence_threshold
        ):
            target_weight = config.low_confidence_target_mix * (config.low_confidence_threshold - component_confidence) / max(
                config.low_confidence_threshold - 0.55,
                1e-6,
            )
            target_weight = float(np.clip(target_weight, 0.0, config.low_confidence_target_mix))
            candidate_array = (1.0 - target_weight) * candidate_array + target_weight * target

        hard_component = dilate_mask(component, config.dilate_radius)
        smooth_component = feather_mask(hard_component, config.feather_radius)
        candidate_array = _harmonize(candidate_array, bg_array, component, config.color_harmonize_strength)
        foreground = smooth_component[..., None] * candidate_array + (1.0 - smooth_component[..., None]) * foreground

        attribution[component > 0.5] = _expert_color(best_name)
        confidence[component > 0.5] = component_confidence
        component_pairs = _region_preference_pairs(
            request.id,
            component_id,
            component_scores,
            margin=config.preference_margin,
        )
        region_preference_pairs.extend(component_pairs)
        selected_components.append(
            {
                "component_id": component_id,
                "expert": best_name,
                "soft_weights": soft_weights,
                "ranking": [
                    {"expert": name, "component_score": float(value), "reward": float(-value)}
                    for value, name in component_scores
                ],
                "preference_pairs": component_pairs,
                "area": int(component.sum()),
                "bbox_xyxy": _component_bbox(component),
                "component_score": float(best_score),
                "component_reward": float(-best_score),
                "second_component_score": float(second_score) if second_score is not None else None,
                "reward_margin_to_second": float(second_score - best_score) if second_score is not None else None,
                "confidence": component_confidence,
            }
        )

    teacher_image, fusion_method = seamless_blend(
        float_to_image(bg_array),
        float_to_image(foreground),
        edit_mask,
        dilate_radius=config.dilate_radius,
        feather_radius=config.feather_radius,
        pyramid_levels=config.pyramid_levels,
        poisson=config.poisson_blend,
        poisson_mode=config.poisson_mode,
    )
    local_refine_metadata = {"enabled": config.local_refiner.enabled, "used": False, "provider": config.local_refiner.provider}
    if config.local_refiner.enabled:
        refiner = LocalRefiner(config.local_refiner)
        debug_dir = Path(output_image_path).parent / "local_refine_debug" / request.id
        teacher_image, local_refine_metadata = refiner.refine(
            source=source_pil,
            fused=teacher_image,
            edit_mask=edit_mask,
            instruction=request.instruction,
            debug_dir=debug_dir,
        )
        if local_refine_metadata.get("used"):
            fusion_method += f"+local_refine_{local_refine_metadata.get('method', config.local_refiner.provider)}"
    canonical_target_image = None
    canonical_fusion_method = None
    if canonical_target_path:
        canonical_target_image, canonical_fusion_method = seamless_blend(
            source_pil,
            target_pil,
            edit_mask,
            dilate_radius=config.dilate_radius,
            feather_radius=config.feather_radius,
            pyramid_levels=config.pyramid_levels,
            poisson=config.poisson_blend,
            poisson_mode=config.poisson_mode,
        )

    teacher_array = image_to_float(teacher_image)
    target_edit_change = _masked_l1(target, source, edit_mask)
    teacher_edit_change = _masked_l1(teacher_array, source, edit_mask)
    edit_change_ratio = float(teacher_edit_change / max(target_edit_change, 1e-8))
    edit_activity_safeguard = {
        "used": False,
        "reason": None,
        "fallback": None,
        "teacher_edit_change_l1": teacher_edit_change,
        "target_edit_change_l1": target_edit_change,
        "edit_change_ratio": edit_change_ratio,
        "min_edit_change_ratio": config.min_edit_change_ratio,
        "min_target_edit_change": config.min_target_edit_change,
    }
    if (
        config.low_edit_fallback != "off"
        and target_edit_change >= config.min_target_edit_change
        and edit_change_ratio < config.min_edit_change_ratio
    ):
        if config.low_edit_fallback == "canonical_target" and canonical_target_image is not None:
            teacher_image = canonical_target_image
            fusion_method += "+low_edit_fallback_canonical_target"
            edit_activity_safeguard.update(
                {
                    "used": True,
                    "reason": "teacher_edit_change_too_low",
                    "fallback": "canonical_target",
                }
            )
        elif config.low_edit_fallback == "target":
            teacher_image = target_pil
            fusion_method += "+low_edit_fallback_target"
            edit_activity_safeguard.update(
                {
                    "used": True,
                    "reason": "teacher_edit_change_too_low",
                    "fallback": "target",
                }
            )
        if edit_activity_safeguard["used"]:
            teacher_array = image_to_float(teacher_image)
            teacher_edit_change = _masked_l1(teacher_array, source, edit_mask)
            edit_change_ratio = float(teacher_edit_change / max(target_edit_change, 1e-8))
            edit_activity_safeguard["teacher_edit_change_l1_after"] = teacher_edit_change
            edit_activity_safeguard["edit_change_ratio_after"] = edit_change_ratio

    teacher_score = score_candidate_arrays(
        source=source,
        candidate=teacher_array,
        target=target,
        edit_mask=edit_mask,
        name="keepedit_release_moe",
        image_path=output_image_path,
        edit_weight=config.edit_weight,
        background_weight=config.background_weight,
        semantic_weight=config.semantic_weight,
        boundary_weight=config.boundary_weight,
    )
    best_expert = min(scores, key=lambda item: item.combined_score)
    fusion_fallback = {
        "used": False,
        "reason": None,
        "fallback": None,
        "teacher_combined_score": float(teacher_score.combined_score),
        "best_expert": best_expert.name,
        "best_expert_combined_score": float(best_expert.combined_score),
        "epsilon": config.fusion_fallback_epsilon,
    }
    if (
        not edit_activity_safeguard["used"]
        and teacher_score.combined_score > best_expert.combined_score + config.fusion_fallback_epsilon
    ):
        if config.fusion_fallback == "best_expert":
            teacher_image = float_to_image(arrays[best_expert.name])
            fusion_method += "+fallback_best_expert"
            save_image(teacher_image, output_image_path)
            fusion_fallback.update(
                {
                    "used": True,
                    "reason": "fused_teacher_worse_than_best_expert",
                    "fallback": best_expert.name,
                }
            )
            teacher_array = image_to_float(teacher_image)
            teacher_score = score_candidate_arrays(
                source=source,
                candidate=teacher_array,
                target=target,
                edit_mask=edit_mask,
                name="keepedit_release_moe",
                image_path=output_image_path,
                edit_weight=config.edit_weight,
                background_weight=config.background_weight,
                semantic_weight=config.semantic_weight,
                boundary_weight=config.boundary_weight,
            )
        elif config.fusion_fallback == "target":
            teacher_image = target_pil
            fusion_method += "+fallback_target"
            save_image(teacher_image, output_image_path)
            fusion_fallback.update(
                {
                    "used": True,
                    "reason": "fused_teacher_worse_than_best_expert",
                    "fallback": "target",
                }
            )
            teacher_array = image_to_float(teacher_image)
            teacher_score = score_candidate_arrays(
                source=source,
                candidate=teacher_array,
                target=target,
                edit_mask=edit_mask,
                name="keepedit_release_moe",
                image_path=output_image_path,
                edit_weight=config.edit_weight,
                background_weight=config.background_weight,
                semantic_weight=config.semantic_weight,
                boundary_weight=config.boundary_weight,
            )
    save_image(teacher_image, output_image_path)
    _save_attribution(Path(attribution_path), attribution)
    _save_confidence(Path(confidence_path), confidence)
    if canonical_target_path:
        if canonical_target_image is not None:
            save_image(canonical_target_image, canonical_target_path)

    if edit_mask.sum() > 0:
        teacher_confidence = float(confidence[edit_mask > 0.5].mean())
    else:
        teacher_confidence = float(np.mean([item.confidence for item in scores]))

    metadata = {
        "stage": "moe_fusion_teacher",
        "algorithm": "keepedit_release_region_preference_calibrated_moe",
        "fusion_strategy": "region_preference_calibrated_soft_routing_laplacian_blend",
        "rank_calibrator": {
            "enabled": config.rank_calibrator.enabled,
            "expert_bias": config.rank_calibrator.expert_bias,
            "edit_mse_weight": config.rank_calibrator.edit_mse_weight,
            "bg_mse_weight": config.rank_calibrator.bg_mse_weight,
            "direction_weight": config.rank_calibrator.direction_weight,
            "boundary_weight": config.rank_calibrator.boundary_weight,
        },
        "fusion_method": fusion_method,
        "fusion_fallback": fusion_fallback,
        "edit_activity_safeguard": edit_activity_safeguard,
        "local_refinement": local_refine_metadata,
        "canonical_target_image": str(canonical_target_path) if canonical_target_path else None,
        "canonical_fusion_method": canonical_fusion_method,
        "selected_bg_expert": selected_bg,
        "selected_edit_experts": selected_components,
        "teacher_confidence": teacher_confidence,
        "teacher_quality_score": float(teacher_score.combined_score),
        "teacher_quality_reward": float(-teacher_score.combined_score),
        "edit_mask_ratio": edit_ratio,
        "global_like": global_like,
        "soft_top_k": config.soft_top_k,
        "temperature": config.temperature,
        "hard_route_confidence": config.hard_route_confidence,
        "region_preference_pairs": region_preference_pairs,
        "per_expert_scores": {item.name: item.to_dict() for item in scores},
    }
    return MoEFusionResult(
        image=teacher_image,
        attribution_map=Image.fromarray(attribution, mode="RGB"),
        confidence_map=Image.fromarray(np.clip(confidence * 255.0, 0, 255).astype(np.uint8), mode="L"),
        scores=scores,
        selected_bg_expert=selected_bg,
        selected_edit_experts=selected_components,
        teacher_confidence=teacher_confidence,
        metadata=metadata,
    )


def persist_mask(mask_path: str | Path, out_path: str | Path) -> Path:
    mask = load_mask(mask_path)
    return save_mask(mask, out_path)
