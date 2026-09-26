"""Датасеты обеих стадий поверх общего индекса групп (см. sources.py)."""
from __future__ import annotations

import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.vehicle_reid.data.sources import Group, Record
from src.vehicle_reid.utils.bbox import BBox, crop_with_padding

logger = logging.getLogger(__name__)

cv2.setNumThreads(0)  # иначе воркеры DataLoader конкурируют за потоки OpenCV


def load_crop(rec: Record, bbox_padding: float = 0.0) -> np.ndarray:
    image = cv2.imread(rec.path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать {rec.path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if rec.bbox is None:
        return image
    x, y, w, h = rec.bbox
    return crop_with_padding(image, BBox(x=x, y=y, w=w, h=h), padding=bbox_padding)


class _CropLoaderMixin:
    bbox_padding: float

    def _load(self, rec: Record) -> np.ndarray:
        return load_crop(rec, self.bbox_padding)


# --------------------------------------------------------------- стадия 1: DINO


class MultiCropDataset(Dataset, _CropLoaderMixin):
    """Один элемент = одна группа: 2 глобальных кропа + N локальных.

    Если в группе несколько изображений, глобальные виды берутся с РАЗНЫХ
    изображений — это даёт «бесплатные» кросс-камерные позитивы.
    Для неразмеченных одиночных картинок поведение сводится к обычному DINO.
    """

    def __init__(self, groups: list[Group], global_transforms, local_transform,
                 n_local_crops: int = 6, bbox_padding: float = 0.0):
        self.groups = groups
        self.global_transforms = global_transforms  # (t1, t2)
        self.local_transform = local_transform
        self.n_local_crops = n_local_crops
        self.bbox_padding = bbox_padding
        if not groups:
            raise ValueError("Пустой список групп")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        recs = self.groups[idx].records
        n = len(recs)

        g_idx = random.sample(range(n), 2) if n >= 2 else [0, 0]
        l_idx = [random.randrange(n) for _ in range(self.n_local_crops)]

        cache: dict[int, np.ndarray] = {}

        def get(i: int) -> np.ndarray:
            if i not in cache:
                cache[i] = self._load(recs[i])
            return cache[i]

        t1, t2 = self.global_transforms
        global_crops = torch.stack([
            t1(image=get(g_idx[0]))["image"],
            t2(image=get(g_idx[1]))["image"],
        ])
        local_crops = torch.stack([
            self.local_transform(image=get(i))["image"] for i in l_idx
        ])
        return {"global": global_crops, "local": local_crops, "index": idx}


# -------------------------------------------------- стадия 2: contrastive/ID


class ReIDTrainDataset(Dataset, _CropLoaderMixin):
    """Один элемент = одно изображение (кроп) с меткой pid.

    n_views=2 возвращает две независимо аугментированные версии — это даёт
    позитивную пару даже когда PK-сэмплер не смог набрать K картинок на ID.
    """

    def __init__(self, groups: list[Group], transform, n_views: int = 2,
                 bbox_padding: float = 0.0, pid_to_label: dict[str, int] | None = None):
        labeled = [g for g in groups if g.pid is not None]
        if not labeled:
            raise ValueError("Для стадии 2 нужны размеченные источники (vehicle_id)")

        pids = sorted({g.pid for g in labeled})
        self.pid_to_label = pid_to_label or {pid: i for i, pid in enumerate(pids)}
        self.transform = transform
        self.n_views = n_views
        self.bbox_padding = bbox_padding

        self.records: list[Record] = []
        self.labels: list[int] = []
        self.pids: list[str] = []
        for g in labeled:
            if g.pid not in self.pid_to_label:
                continue
            for rec in g.records:
                self.records.append(rec)
                self.labels.append(self.pid_to_label[g.pid])
                self.pids.append(g.pid)
        self.labels_np = np.asarray(self.labels, dtype=np.int64)

    @property
    def num_classes(self) -> int:
        return len(self.pid_to_label)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        image = self._load(self.records[idx])
        views = torch.stack([self.transform(image=image)["image"] for _ in range(self.n_views)])
        return {"views": views, "label": self.labels[idx], "index": idx}


class ReIDEvalDataset(Dataset, _CropLoaderMixin):
    """Один элемент = один кроп, без аугментаций. Используется для mAP/CMC и инференса."""

    def __init__(self, groups: list[Group], transform, bbox_padding: float = 0.0):
        self.transform = transform
        self.bbox_padding = bbox_padding
        self.records: list[Record] = []
        self.pids: list[str] = []
        for g in groups:
            for rec in g.records:
                self.records.append(rec)
                self.pids.append(g.pid if g.pid is not None else "")
        self.pids_np = np.asarray(self.pids)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        image = self._load(self.records[idx])
        tensor = self.transform(image=image)["image"]
        return {"image": tensor, "pid": self.pids[idx], "path": self.records[idx].path, "index": idx}


# ------------------------------------------------------------------- сплиты


def split_groups_by_id(groups: list[Group], val_ratio: float, seed: int = 42
                       ) -> tuple[list[Group], list[Group]]:
    """Разделение по ID (disjoint identities) — честная оценка обобщения на новые машины."""
    labeled = [g for g in groups if g.pid is not None]
    rng = random.Random(seed)
    pids = sorted({g.pid for g in labeled})
    rng.shuffle(pids)
    n_val = int(round(len(pids) * val_ratio))
    val_pids = set(pids[:n_val])
    train = [g for g in labeled if g.pid not in val_pids]
    val = [g for g in labeled if g.pid in val_pids]
    logger.info("Сплит по ID: train %d ID / val %d ID", len(pids) - len(val_pids), len(val_pids))
    return train, val


def make_query_gallery(groups: list[Group], seed: int = 42, n_query_per_id: int = 1
                       ) -> tuple[list[Group], list[Group]]:
    """Из валидационных групп делает query/gallery: по n картинок на ID в query, остальное в gallery.

    ID, у которых меньше 2 изображений, отбрасываются — по ним невозможно посчитать retrieval.
    """
    rng = random.Random(seed)
    query: list[Group] = []
    gallery: list[Group] = []
    for g in groups:
        if len(g.records) < 2:
            continue
        recs = list(g.records)
        rng.shuffle(recs)
        k = min(n_query_per_id, len(recs) - 1)
        query.append(Group(records=recs[:k], pid=g.pid, source=g.source))
        gallery.append(Group(records=recs[k:], pid=g.pid, source=g.source))
    if not query:
        raise ValueError("Не удалось построить query/gallery: у всех ID меньше 2 изображений")
    return query, gallery
