from __future__ import annotations


class BBoxValidationError(ValueError):
    pass


import math
from dataclasses import dataclass



@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    def validate(self, img_w: int, img_h: int, min_size: int = 8):
        if any(v is None for v in (self.x, self.y, self.w, self.h)):
            raise BBoxValidationError("BBox содержит пустые значения")
        if self.w <= 0 or self.h <= 0:
            raise BBoxValidationError("Ширина и высота BBox должны быть > 0")
        if self.w < min_size or self.h < min_size:
            raise BBoxValidationError(f"BBox слишком мал (< {min_size}px)")
        if self.x < 0 or self.y < 0:
            raise BBoxValidationError("Координаты BBox не могут быть отрицательными")
        if self.x + self.w > img_w + 1 or self.y + self.h > img_h + 1:
            raise BBoxValidationError("BBox выходит за границы изображения")

    def to_xyxy(self, img_w: int, img_h: int, patch_size: int = 0) -> tuple[int, int, int, int]:
        """Целочисленные координаты, обрезанные по границам изображения.
        patch_size > 0 — стороны расширяются до кратных patch_size (если влезает в картинку)."""
        x1 = max(0, math.floor(self.x))
        y1 = max(0, math.floor(self.y))
        x2 = min(img_w, math.ceil(self.x + self.w))
        y2 = min(img_h, math.ceil(self.y + self.h))
        if x2 <= x1 or y2 <= y1:
            raise BBoxValidationError(f"Пустой bbox после обрезки: {self}")
        if patch_size > 0:
            x1, x2 = self._snap(x1, x2, img_w, patch_size)
            y1, y2 = self._snap(y1, y2, img_h, patch_size)
        return x1, y1, x2, y2

    @staticmethod
    def _snap(lo: int, hi: int, limit: int, p: int) -> tuple[int, int]:
        target = math.ceil((hi - lo) / p) * p
        if target > limit:
            target = (limit // p) * p
        if target == 0:
            return lo, hi
        lo = min(lo, limit - target)  # если не влезает вправо — сдвигаем влево
        return lo, lo + target

def crop_with_padding(image, bbox: "BBox", padding: float = 0.0, patch_size: int = 0):
    """Кроп по bbox с относительным расширением рамки на padding с каждой стороны.

    image: np.ndarray [H, W, 3]. Возвращает непрерывный массив (copy-free там, где можно).
    """
    import numpy as np

    img_h, img_w = image.shape[:2]
    if padding > 0:
        pad_w = bbox.w * padding
        pad_h = bbox.h * padding
        bbox = BBox(
            x=bbox.x - pad_w,
            y=bbox.y - pad_h,
            w=bbox.w + 2 * pad_w,
            h=bbox.h + 2 * pad_h,
        )
    x1, y1, x2, y2 = bbox.to_xyxy(img_w=img_w, img_h=img_h, patch_size=patch_size)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise BBoxValidationError(f"Пустой кроп для bbox={bbox}")
    return np.ascontiguousarray(crop)
