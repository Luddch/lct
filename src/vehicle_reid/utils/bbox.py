from __future__ import annotations
import numpy as np
from dataclasses import dataclass


class BBoxValidationError(ValueError):
    pass


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

    def to_xyxy(self):
        return self.x, self.y, self.x + self.w, self.y + self.h


def crop_with_padding(image: np.ndarray, bbox: BBox, padding_ratio: float = 0.1) -> np.ndarray:
    """Вырезает область BBox с относительным паддингом и клипом по границам."""
    img_h, img_w = image.shape[:2]
    bbox.validate(img_w, img_h)

    pad_w = bbox.w * padding_ratio
    pad_h = bbox.h * padding_ratio

    x1 = int(max(0, round(bbox.x - pad_w)))
    y1 = int(max(0, round(bbox.y - pad_h)))
    x2 = int(min(img_w, round(bbox.x + bbox.w + pad_w)))
    y2 = int(min(img_h, round(bbox.y + bbox.h + pad_h)))

    if x2 <= x1 or y2 <= y1:
        raise BBoxValidationError("Некорректный BBox после обрезки")

    return image[y1:y2, x1:x2].copy()
