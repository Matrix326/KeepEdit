from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters, morphology, transform

from keepedit.io import ensure_dir


def load_rgb(path: str | Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def save_image(image: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    image.save(path)
    return path


def image_to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def float_to_image(array: np.ndarray) -> Image.Image:
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def load_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if size and mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return (np.asarray(mask, dtype=np.float32) / 255.0).clip(0.0, 1.0)


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    mask = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(path)
    return path


def resize_mask(mask: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == size_hw:
        return mask.astype(np.float32)
    return transform.resize(
        mask,
        size_hw,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)


def full_mask(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.ones((height, width), dtype=np.float32)


def diff_mask(
    input_image: Image.Image,
    target_image: Image.Image,
    threshold: float = 0.06,
    min_area_ratio: float = 0.002,
) -> np.ndarray:
    target_image = target_image.resize(input_image.size, Image.Resampling.LANCZOS)
    x = image_to_float(input_image)
    y = image_to_float(target_image)
    diff = np.abs(y - x).mean(axis=2)
    mask = diff > threshold

    min_area = int(mask.size * min_area_ratio)
    cleaned = morphology.remove_small_objects(mask, min_size=max(4, min_area))
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=max(4, min_area))
    if cleaned.sum() == 0:
        cutoff = np.quantile(diff, 0.95)
        cleaned = diff >= max(cutoff, threshold * 0.5)
    return cleaned.astype(np.float32)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0.5).astype(np.float32)
    return morphology.binary_dilation(mask > 0.5, morphology.disk(radius)).astype(np.float32)


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0.5).astype(np.float32)
    blurred = filters.gaussian(mask.astype(np.float32), sigma=radius, preserve_range=True)
    max_value = float(blurred.max())
    if max_value > 0:
        blurred = blurred / max_value
    return blurred.clip(0.0, 1.0).astype(np.float32)


def edit_band(mask: np.ndarray, radius: int = 6) -> np.ndarray:
    outer = dilate_mask(mask, radius)
    inner = morphology.binary_erosion(mask > 0.5, morphology.disk(max(1, radius))).astype(np.float32)
    return np.clip(outer - inner, 0.0, 1.0)


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).clip(0.0, 1.0)


def mask_boundary_band(mask: np.ndarray, radius: int = 6) -> np.ndarray:
    hard = mask > 0.5
    if hard.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    outer = morphology.binary_dilation(hard, morphology.disk(max(1, radius)))
    inner = morphology.binary_erosion(hard, morphology.disk(max(1, radius)))
    return (outer & ~inner).astype(np.float32)


def laplacian_pyramid_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: np.ndarray,
    levels: int = 5,
) -> np.ndarray:
    """Multi-band blend foreground into background using an alpha mask.

    Inputs are float RGB arrays in [0, 1] and a float mask in [0, 1].
    """

    levels = max(1, int(levels))
    alpha = alpha.astype(np.float32)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    bg_pyr = [background.astype(np.float32)]
    fg_pyr = [foreground.astype(np.float32)]
    mask_pyr = [alpha.astype(np.float32)]

    for _ in range(levels - 1):
        h, w = bg_pyr[-1].shape[:2]
        if min(h, w) <= 16:
            break
        size = (max(1, h // 2), max(1, w // 2))
        bg_pyr.append(transform.resize(bg_pyr[-1], (*size, 3), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32))
        fg_pyr.append(transform.resize(fg_pyr[-1], (*size, 3), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32))
        mask_pyr.append(transform.resize(mask_pyr[-1], (*size, 1), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32).clip(0, 1))

    bg_lap = []
    fg_lap = []
    for level in range(len(bg_pyr) - 1):
        target_shape = bg_pyr[level].shape
        bg_up = transform.resize(bg_pyr[level + 1], target_shape, order=1, anti_aliasing=False, preserve_range=True).astype(np.float32)
        fg_up = transform.resize(fg_pyr[level + 1], target_shape, order=1, anti_aliasing=False, preserve_range=True).astype(np.float32)
        bg_lap.append(bg_pyr[level] - bg_up)
        fg_lap.append(fg_pyr[level] - fg_up)
    bg_lap.append(bg_pyr[-1])
    fg_lap.append(fg_pyr[-1])

    blended = mask_pyr[-1] * fg_lap[-1] + (1.0 - mask_pyr[-1]) * bg_lap[-1]
    for level in range(len(bg_lap) - 2, -1, -1):
        blended = transform.resize(blended, bg_lap[level].shape, order=1, anti_aliasing=False, preserve_range=True).astype(np.float32)
        mask = mask_pyr[level]
        blended = blended + mask * fg_lap[level] + (1.0 - mask) * bg_lap[level]
    return blended.clip(0.0, 1.0)


def poisson_blend_opencv(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
    mode: str = "mixed",
) -> np.ndarray | None:
    """Optional OpenCV seamlessClone fallback for gradient-domain harmonization."""

    try:
        import cv2
    except Exception:
        return None
    hard = mask > 0.5
    if hard.sum() <= 4:
        return None
    ys, xs = np.where(hard)
    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    src = np.clip(foreground * 255.0, 0, 255).astype(np.uint8)
    dst = np.clip(background * 255.0, 0, 255).astype(np.uint8)
    mask_u8 = (hard.astype(np.uint8) * 255)
    clone_flag = cv2.MIXED_CLONE if mode == "mixed" else cv2.NORMAL_CLONE
    try:
        blended = cv2.seamlessClone(src, dst, mask_u8, center, clone_flag)
    except Exception:
        return None
    return (blended.astype(np.float32) / 255.0).clip(0.0, 1.0)


def seamless_blend(
    input_image: Image.Image,
    edited_image: Image.Image,
    mask: np.ndarray,
    dilate_radius: int = 8,
    feather_radius: int = 12,
    pyramid_levels: int = 5,
    poisson: bool = False,
    poisson_mode: str = "mixed",
) -> tuple[Image.Image, str]:
    """Blend an edited image into an input image using feathering and multi-band blending."""

    edited_image = edited_image.resize(input_image.size, Image.Resampling.LANCZOS)
    edit_region = dilate_mask(resize_mask(mask, (input_image.height, input_image.width)), dilate_radius)
    alpha = feather_mask(edit_region, feather_radius)
    x = image_to_float(input_image)
    y = image_to_float(edited_image)
    blended = laplacian_pyramid_blend(x, y, alpha, levels=pyramid_levels)
    method = f"laplacian_pyramid_l{pyramid_levels}_feather"
    if poisson:
        poisson_result = poisson_blend_opencv(x, blended, edit_region, mode=poisson_mode)
        if poisson_result is not None:
            # Preserve multi-band low-frequency smoothness and use Poisson only as
            # a gradient-domain harmonizer inside the confident edit region.
            hard_alpha = feather_mask(edit_region, max(1, feather_radius // 2))
            blended = hard_alpha[..., None] * poisson_result + (1.0 - hard_alpha[..., None]) * blended
            method += f"+poisson_{poisson_mode}"
    return float_to_image(blended), method


def hard_copy_background(
    input_image: Image.Image,
    edited_image: Image.Image,
    mask: np.ndarray,
    dilate_radius: int = 8,
    feather_radius: int = 12,
) -> Image.Image:
    edited_image = edited_image.resize(input_image.size, Image.Resampling.LANCZOS)
    edit_region = dilate_mask(resize_mask(mask, (input_image.height, input_image.width)), dilate_radius)
    edit_region = feather_mask(edit_region, feather_radius)
    x = image_to_float(input_image)
    y = image_to_float(edited_image)
    composed = (1.0 - edit_region[..., None]) * x + edit_region[..., None] * y
    return float_to_image(composed)


def difference_heatmap(input_image: Image.Image, output_image: Image.Image) -> np.ndarray:
    output_image = output_image.resize(input_image.size, Image.Resampling.LANCZOS)
    return np.abs(image_to_float(output_image) - image_to_float(input_image)).mean(axis=2)
