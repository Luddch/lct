"""Единое описание источников данных для обеих стадий обучения.

Стадия 1 (DINO) работает со списком «групп»: группа — это набор кропов,
про которые мы считаем, что они изображают один и тот же объект.
Для размеченных источников группа = vehicle_id, для неразмеченных —
одна картинка (или папка, если в датасете есть треки).

Стадия 2 (contrastive) использует тот же индекс, но с обязательными метками.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

REQUIRED_CSV_COLUMNS = {"image_id"}
BBOX_COLUMNS = ("x", "y", "w", "h")


@dataclass(frozen=True)
class Record:
    """Один кроп: путь к картинке + опциональный bbox (x, y, w, h) в пикселях."""

    path: str
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class Group:
    """Набор кропов одного объекта. pid не None только для размеченных данных."""

    records: list[Record]
    pid: str | None = None
    source: str = ""


@dataclass
class SourceSpec:
    """Описание одного датасета в конфиге.

    type:
      folder — рекурсивный обход images_dir, bbox отсутствует (готовые кропы);
      csv    — csv с колонкой image_id (+ опционально x,y,w,h и vehicle_id).
    group_by:
      none   — каждая картинка отдельная группа (классический DINO);
      parent — группировка по имени родительской папки (треки/ID-папки);
      id     — группировка по колонке id_column из csv.
    """

    name: str
    type: Literal["folder", "csv"] = "folder"
    images_dir: str = ""
    csv: str | None = None
    group_by: Literal["none", "parent", "id"] = "none"
    id_column: str = "vehicle_id"
    image_column: str = "image_id"
    image_extension: str = ".jpg"
    use_bbox: bool = True
    repeat: int = 1  # сколько раз источник повторяется в эпохе (простое взвешивание)
    limit: int | None = None  # ограничение числа групп, удобно для отладки
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS

    def __post_init__(self):
        if self.type == "csv" and not self.csv:
            raise ValueError(f"Источник '{self.name}': type=csv требует поле csv")
        if self.type == "folder" and self.group_by == "id":
            raise ValueError(f"Источник '{self.name}': group_by=id доступен только для csv")
        if self.repeat < 1:
            raise ValueError(f"Источник '{self.name}': repeat должен быть >= 1")


# --------------------------------------------------------------------- folder


def _iter_images(root: Path, extensions: Iterable[str]) -> list[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def groups_from_folder(spec: SourceSpec) -> list[Group]:
    root = Path(spec.images_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Источник '{spec.name}': нет директории {root}")

    paths = _iter_images(root, spec.extensions)
    if not paths:
        raise ValueError(f"Источник '{spec.name}': в {root} не найдено изображений")

    if spec.group_by == "parent":
        buckets: dict[str, list[Record]] = {}
        for p in paths:
            buckets.setdefault(p.parent.name, []).append(Record(path=str(p)))
        groups = [Group(records=recs, source=spec.name) for _, recs in sorted(buckets.items())]
    else:
        groups = [Group(records=[Record(path=str(p))], source=spec.name) for p in paths]
    return groups


# ------------------------------------------------------------------------ csv


def _resolve_path(images_dir: Path, image_id: str, extension: str) -> str:
    candidate = images_dir / image_id
    if candidate.suffix:
        return str(candidate)
    return str(candidate.with_suffix(extension))


def groups_from_csv(spec: SourceSpec, require_pid: bool = False) -> list[Group]:
    df = pd.read_csv(spec.csv, dtype={spec.image_column: str, spec.id_column: str})
    if spec.image_column not in df.columns:
        raise ValueError(f"Источник '{spec.name}': в csv нет колонки {spec.image_column}")

    has_bbox = spec.use_bbox and all(c in df.columns for c in BBOX_COLUMNS)
    if spec.use_bbox and not has_bbox:
        logger.warning("Источник '%s': колонок bbox нет, используются целые картинки", spec.name)
    if has_bbox:
        df = df.dropna(subset=list(BBOX_COLUMNS))

    has_pid = spec.id_column in df.columns
    if require_pid and not has_pid:
        raise ValueError(f"Источник '{spec.name}': нужна колонка {spec.id_column}")
    if spec.group_by == "id" and not has_pid:
        raise ValueError(f"Источник '{spec.name}': group_by=id, но колонки {spec.id_column} нет")

    images_dir = Path(spec.images_dir)

    def to_record(row) -> Record:
        bbox = None
        if has_bbox:
            bbox = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
        return Record(path=_resolve_path(images_dir, str(row[spec.image_column]), spec.image_extension), bbox=bbox)

    groups: list[Group] = []
    if spec.group_by == "id" or (require_pid and has_pid):
        for pid, chunk in df.groupby(spec.id_column, sort=True):
            recs = [to_record(r) for _, r in chunk.iterrows()]
            groups.append(Group(records=recs, pid=str(pid), source=spec.name))
    else:
        for _, row in df.iterrows():
            groups.append(Group(records=[to_record(row)], source=spec.name))
    return groups


# ---------------------------------------------------------------------- build


def build_groups(specs: list[SourceSpec], require_pid: bool = False) -> list[Group]:
    """Собирает группы со всех источников, применяя limit и repeat."""
    all_groups: list[Group] = []
    for spec in specs:
        groups = groups_from_csv(spec, require_pid) if spec.type == "csv" else groups_from_folder(spec)
        if spec.limit is not None:
            groups = groups[: spec.limit]
        n_images = sum(len(g.records) for g in groups)
        logger.info(
            "Источник '%s': %d групп, %d изображений (repeat=%d)",
            spec.name, len(groups), n_images, spec.repeat,
        )
        all_groups.extend(groups * spec.repeat)
    if not all_groups:
        raise ValueError("Не найдено ни одной группы — проверьте секцию sources в конфиге")
    logger.info("Всего: %d групп, %d изображений", len(all_groups),
                sum(len(g.records) for g in all_groups))
    return all_groups


def specs_from_config(raw: list[dict]) -> list[SourceSpec]:
    specs = [SourceSpec(**item) for item in raw]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Имена источников должны быть уникальны: {names}")
    return specs


@dataclass
class DatasetStats:
    n_groups: int
    n_images: int
    per_source: dict[str, int] = field(default_factory=dict)


def describe(groups: list[Group]) -> DatasetStats:
    per_source: dict[str, int] = {}
    for g in groups:
        per_source[g.source] = per_source.get(g.source, 0) + len(g.records)
    return DatasetStats(
        n_groups=len(groups),
        n_images=sum(len(g.records) for g in groups),
        per_source=per_source,
    )
