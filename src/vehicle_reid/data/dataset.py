from __future__ import annotations
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.vehicle_reid.utils.bbox import BBox, crop_with_padding


class VehicleReIDDataset(Dataset):
    """
    Универсальный датасет:
      - train режим: возвращает (image, pid_label, [attr_labels])
      - query/gallery режим: возвращает (image, image_id)
    """

    def __init__(self, csv_path, images_dir, transforms=None,
                 padding_ratio=0.1, mode="train",
                 pid_to_label=None, attribute_columns=None,
                 attribute_encoders=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transforms = transforms
        self.padding_ratio = padding_ratio
        self.mode = mode
        self.attribute_columns = attribute_columns or []
        self.attribute_encoders = attribute_encoders or {}

        if mode == "train":
            assert "vehicle_id" in self.df.columns
            if pid_to_label is None:
                unique_ids = sorted(self.df["vehicle_id"].unique().tolist())
                pid_to_label = {pid: i for i, pid in enumerate(unique_ids)}
            self.pid_to_label = pid_to_label
            self.num_classes = len(self.pid_to_label)
        else:
            self.pid_to_label = None

    def __len__(self):
        return len(self.df)

    def _load_crop(self, row):
        img_path = os.path.join(self.images_dir, str(row["image_id"]))
        if not os.path.isfile(img_path):
            # допускаем отсутствие расширения в id
            for ext in (".jpg", ".jpeg", ".png"):
                cand = img_path + ext
                if os.path.isfile(cand):
                    img_path = cand
                    break
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox = BBox(x=float(row["x"]), y=float(row["y"]), w=float(row["w"]), h=float(row["h"]))
        crop = crop_with_padding(image, bbox, self.padding_ratio)
        return crop

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop = self._load_crop(row)

        if self.transforms is not None:
            crop = self.transforms(image=crop)["image"]

        if self.mode == "train":
            label = self.pid_to_label[row["vehicle_id"]]
            item = {"image": crop, "label": label, "image_id": row["image_id"]}
            for col in self.attribute_columns:
                enc = self.attribute_encoders[col]
                item[f"attr_{col}"] = enc[row[col]]
            return item
        else:
            return {"image": crop, "image_id": row["image_id"]}
