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