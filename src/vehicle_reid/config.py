"""Загрузка YAML-конфигов стадий обучения и инференса."""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, List, Type, TypeVar

import yaml

from src.vehicle_reid.data.sources import SourceSpec, specs_from_config

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _instantiate(cls: Type[T], data: dict[str, Any]) -> T:
    """Создаёт dataclass из словаря, ругаясь на неизвестные ключи (защита от опечаток)."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: неизвестные поля в конфиге: {sorted(unknown)}")
    return cls(**data)


def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_stage_config(path: str, cls: Type[T]) -> tuple[T, list[SourceSpec]]:
    """Читает конфиг вида {sources: [...], <остальные поля датакласса>}."""
    raw = read_yaml(path)
    sources = specs_from_config(raw.pop("sources", []))
    # секции допускаются для читаемости: их содержимое просто сливается
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return _instantiate(cls, flat), sources


# ------------------------------------------------------ конфиг сервиса/поиска


@dataclass
class ServingConfig:
    weights: str = "checkpoints/reid/best.pt"
    backbone: str = "vit_large_patch16_dinov3.lvd1689m"
    embedding_dim: int = 0
    pooling: str = "cls"
    neck: str = "bnneck"
    image_size: List[int] = field(default_factory=lambda: [224, 224])
    bbox_padding: float = 0.10
    device: str = "cuda"
    metric: str = "cosine"
    backend: str = "torch"
    top_k: int = 10
    reject_threshold: float = 0.35


def load_serving_config(path: str) -> ServingConfig:
    raw = read_yaml(path)
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return _instantiate(ServingConfig, flat)
