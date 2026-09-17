from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataCfg:
    root: str
    images_dir: str
    train_csv: str
    query_csv: str
    gallery_csv: str
    image_size: List[int]
    bbox_padding: float
    val_ratio: float


@dataclass
class ModelCfg:
    backbone: str
    pretrained: bool
    embedding_dim: int
    pooling: str
    neck: str
    use_attributes: bool
    attribute_columns: List[str]


@dataclass
class LossCfg:
    id_loss: str
    arcface_scale: float
    arcface_margin: float
    label_smoothing: float
    triplet_margin: float
    triplet_weight: float
    id_weight: float
    center_weight: float


@dataclass
class TrainCfg:
    epochs: int
    batch_size_p: int
    batch_size_k: int
    num_workers: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    scheduler: str
    amp: bool
    eval_every: int
    early_stop_patience: int


@dataclass
class SearchCfg:
    metric: str
    top_k: int
    reject_threshold: float
    backend: str


@dataclass
class ProjectCfg:
    seed: int
    device: str
    output_dir: str
    weights_dir: str


@dataclass
class Config:
    project: ProjectCfg
    data: DataCfg
    model: ModelCfg
    loss: LossCfg
    train: TrainCfg
    search: SearchCfg


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        project=ProjectCfg(**raw["project"]),
        data=DataCfg(**raw["data"]),
        model=ModelCfg(**raw["model"]),
        loss=LossCfg(**raw["loss"]),
        train=TrainCfg(**raw["train"]),
        search=SearchCfg(**raw["search"]),
    )
