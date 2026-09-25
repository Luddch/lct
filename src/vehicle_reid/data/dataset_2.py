import cv2
import pandas as pd
import torch
from albumentations import ToTensorV2
from torch.utils.data import Dataset
import albumentations as A
from src.vehicle_reid.utils.bbox import BBox

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.vehicle_reid.utils.bbox import BBox

REQUIRED_COLUMNS = {"vehicle_id", "image_id", "x", "y", "w", "h"}


class VehicleReIDDataset(Dataset):
    """Один элемент = одна машина.
    Возвращает 2 глобальных кропа (по возможности с РАЗНЫХ фото этой машины)
    и n_local_crops локальных кропов со случайных фото этой машины.
    Все аугментации выполняются здесь, т.е. в воркерах DataLoader."""

    def __init__(self, csv_path, images_dir, global_transforms, local_transform,
                 n_local_crops: int = 6, min_images: int = 1):
        df = pd.read_csv(csv_path, dtype={"image_id": str, "vehicle_id": str})
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"В CSV нет колонок: {missing}")
        df = df.dropna(subset=list(REQUIRED_COLUMNS))

        self.images_dir = Path(images_dir)
        self.global_transforms = global_transforms  # (t1, t2)
        self.local_transform = local_transform
        self.n_local_crops = n_local_crops

        # Группируем один раз, а не фильтруем весь DataFrame на каждом __getitem__
        self.vehicle_ids: list[str] = []
        self.records: list[list[dict]] = []
        for vid, group in df.groupby("vehicle_id", sort=True):
            recs = group[["image_id", "x", "y", "w", "h"]].to_dict("records")
            if len(recs) >= min_images:
                self.vehicle_ids.append(vid)
                self.records.append(recs)
        if not self.records:
            raise ValueError("Датасет пуст")

    def __len__(self):
        return len(self.records)

    def _load_crop(self, rec: dict) -> np.ndarray:
        path = self.images_dir / f"{rec['image_id']}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Не удалось прочитать {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        bbox = BBox(x=float(rec["x"]), y=float(rec["y"]), w=float(rec["w"]), h=float(rec["h"]))
        x1, y1, x2, y2 = bbox.to_xyxy(img_w=w, img_h=h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"Пустой кроп: {path}, bbox={bbox}")
        return np.ascontiguousarray(crop)

    def __getitem__(self, idx: int):
        recs = self.records[idx]
        n = len(recs)

        # два глобальных вида — с разных фото (кросс-камерные позитивы)
        g_idx = random.sample(range(n), 2) if n >= 2 else [0, 0]
        l_idx = [random.randrange(n) for _ in range(self.n_local_crops)]

        cache: dict[int, np.ndarray] = {}

        def get(i):
            if i not in cache:
                cache[i] = self._load_crop(recs[i])
            return cache[i]

        t1, t2 = self.global_transforms
        global_crops = torch.stack([
            t1(image=get(g_idx[0]))["image"],
            t2(image=get(g_idx[1]))["image"],
        ])                                                     # [2, 3, G, G]
        local_crops = torch.stack([
            self.local_transform(image=get(i))["image"] for i in l_idx
        ])                                                     # [L, 3, S, S]
        return {"global": global_crops, "local": local_crops, "label": idx}
