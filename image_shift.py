"""
Синтез кадров: чистый фон (inpaint) + объект по маске, сдвиг по X, прогон OpenVLA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class ObjectGeometry:
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1
    centroid: Tuple[float, float]
    area_px: int


@dataclass
class ShiftResult:
    dx_px: int
    image: Image.Image
    action: np.ndarray


@dataclass
class ObjectLayer:
    rgba: Image.Image
    bbox: Tuple[int, int, int, int]


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Маска пустая — объект не найден")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def object_geometry(mask: np.ndarray) -> ObjectGeometry:
    mask = np.asarray(mask, dtype=bool)
    x0, y0, x1, y1 = bbox_from_mask(mask)
    ys, xs = np.where(mask)
    return ObjectGeometry(
        bbox=(x0, y0, x1, y1),
        centroid=(float(xs.mean()), float(ys.mean())),
        area_px=int(mask.sum()),
    )


def mask_from_binary(
    mask_image: Union[Image.Image, str, Path],
    *,
    threshold: int = 127,
) -> np.ndarray:
    """Ч/б маска: белые пиксели = объект."""
    if isinstance(mask_image, (str, Path)):
        mask_image = Image.open(mask_image)
    gray = np.array(mask_image.convert("L"))
    mask = gray > threshold
    if mask.sum() < 50:
        raise ValueError("Маска пустая — проверьте image_mask.png")
    return mask


def load_mask(
    original: Union[Image.Image, str, Path],
    mask_path: Union[str, Path],
) -> Tuple[np.ndarray, ObjectGeometry]:
    if isinstance(original, (str, Path)):
        original = Image.open(original)
    mask = mask_from_binary(mask_path)
    if mask.shape[:2] != (original.size[1], original.size[0]):
        raise ValueError(
            f"Маска {mask.shape[:2]} ≠ кадр {original.size[1]}×{original.size[0]}"
        )
    return mask, object_geometry(mask)


def extract_object_layer(
    original: Union[Image.Image, str, Path],
    mask: np.ndarray,
) -> ObjectLayer:
    if isinstance(original, (str, Path)):
        original = Image.open(original)
    mask = np.asarray(mask, dtype=bool)
    x0, y0, x1, y1 = bbox_from_mask(mask)
    rgb = np.array(original.convert("RGB"))
    alpha = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    rgba = np.dstack([rgb[y0:y1, x0:x1], alpha])
    return ObjectLayer(Image.fromarray(rgba, mode="RGBA"), (x0, y0, x1, y1))


def composite_on_background(
    background: Union[Image.Image, str, Path],
    layer: ObjectLayer,
    dx: int = 0,
) -> Image.Image:
    if isinstance(background, (str, Path)):
        background = Image.open(background)
    bg = background.convert("RGB")
    w, h = bg.size
    x0, y0, x1, y1 = layer.bbox
    nx0 = x0 + dx
    if nx0 < 0 or x1 + dx > w or y1 > h or y0 < 0:
        raise ValueError(f"dx={dx} выходит за границы кадра ({w}×{h})")
    out = bg.copy()
    out.paste(layer.rgba, (nx0, y0), layer.rgba)
    return out


def prepare_shift_on_background(
    original: Union[Image.Image, str, Path],
    mask_path: Union[str, Path],
    background: Union[Image.Image, str, Path],
) -> Tuple[Image.Image, ObjectLayer, ObjectGeometry]:
    if isinstance(original, (str, Path)):
        original = Image.open(original)
    if isinstance(background, (str, Path)):
        background = Image.open(background)
    if background.size != original.size:
        raise ValueError(f"Фон {background.size} ≠ оригинал {original.size}")
    mask, geo = load_mask(original, mask_path)
    layer = extract_object_layer(original, mask)
    return background, layer, geo


def make_dx_range(
    dx_min: int,
    dx_max: int,
    step: int,
    *,
    image_width: int,
    bbox: Tuple[int, int, int, int],
) -> List[int]:
    x0, _, x1, _ = bbox
    lo = max(dx_min, -x0)
    hi = min(dx_max, image_width - x1)
    return list(range(lo, hi + 1, step))


def run_shift_sweep(
    vla,
    processor,
    dx_values: Iterable[int],
    prompt: str,
    *,
    background: Union[Image.Image, str, Path],
    object_layer: ObjectLayer,
    device: str = "cuda:0",
    unnorm_key: str = "bridge_orig",
    save_dir: Optional[Union[str, Path]] = None,
) -> List[ShiftResult]:
    import torch

    if isinstance(background, (str, Path)):
        background = Image.open(background)

    results: List[ShiftResult] = []
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for dx in dx_values:
            shifted = composite_on_background(background, object_layer, dx)
            inputs = processor(prompt, shifted).to(device, dtype=torch.bfloat16)
            action = np.asarray(
                vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            )
            results.append(ShiftResult(dx_px=dx, image=shifted, action=action))
            if save_dir is not None:
                shifted.save(Path(save_dir) / f"dx_{dx:+04d}.png")
    return results


def results_to_arrays(results: List[ShiftResult]) -> Tuple[np.ndarray, np.ndarray]:
    dx = np.array([r.dx_px for r in results], dtype=np.int32)
    actions = np.stack([r.action for r in results], axis=0)
    return dx, actions
