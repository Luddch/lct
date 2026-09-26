"""Стадия 2: contrastive fine-tuning на размеченном датасете.

    python -m scripts.train_reid --config configs/reid.yaml \
        --dino-weights checkpoints/dino/teacher_backbone.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.vehicle_reid.config import load_stage_config
from src.vehicle_reid.data.datasets import (
    ReIDEvalDataset, ReIDTrainDataset, make_query_gallery, split_groups_by_id,
)
from src.vehicle_reid.data.sampler import PKSampler
from src.vehicle_reid.data.sources import build_groups
from src.vehicle_reid.data.transforms import build_eval_transforms, build_train_transforms
from src.vehicle_reid.engine.reid_config import ReIDTrainingConfig
from src.vehicle_reid.engine.reid_trainer import ReIDTrainer
from src.vehicle_reid.models.model import VehicleReIDModel

logger = logging.getLogger("train_reid")


def parse_args():
    p = argparse.ArgumentParser("Contrastive fine-tuning for vehicle ReID")
    p.add_argument("--config", default="configs/reid.yaml")
    p.add_argument("--dino-weights", default=None, help="перекрывает dino_weights из конфига")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume", default=None, help="путь к чекпоинту или 'auto'")
    p.add_argument("--no-dino", action="store_true", help="обучить baseline без весов стадии 1")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: ReIDTrainingConfig, sources, device):
    groups = build_groups(sources, require_pid=True)
    train_groups, val_groups = split_groups_by_id(groups, cfg.val_ratio, cfg.seed)
    query_groups, gallery_groups = make_query_gallery(val_groups, cfg.seed, cfg.n_query_per_id)

    train_ds = ReIDTrainDataset(
        train_groups,
        transform=build_train_transforms(cfg.image_size, cfg.erasing_prob),
        n_views=cfg.n_views,
        bbox_padding=cfg.bbox_padding,
    )
    eval_tf = build_eval_transforms(cfg.image_size)
    query_ds = ReIDEvalDataset(query_groups, eval_tf, cfg.bbox_padding)
    gallery_ds = ReIDEvalDataset(gallery_groups, eval_tf, cfg.bbox_padding)

    logger.info("Train: %d изображений / %d ID; val query %d, gallery %d",
                len(train_ds), train_ds.num_classes, len(query_ds), len(gallery_ds))

    pin = device.type == "cuda"
    sampler = (PKSampler(train_ds, cfg.p_per_batch, cfg.k_per_id, cfg.seed)
               if cfg.p_per_batch else None)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.micro_batch_size, sampler=sampler,
        shuffle=sampler is None, drop_last=True, num_workers=cfg.num_workers,
        pin_memory=pin, persistent_workers=cfg.num_workers > 0,
    )
    query_loader = DataLoader(query_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=pin)
    gallery_loader = DataLoader(gallery_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=pin)

    split_info = {
        "train_images": len(train_ds), "num_classes": train_ds.num_classes,
        "query_images": len(query_ds), "gallery_images": len(gallery_ds),
        "val_ratio": cfg.val_ratio, "seed": cfg.seed,
    }
    return train_loader, query_loader, gallery_loader, train_ds.num_classes, split_info


def build_model(cfg: ReIDTrainingConfig, num_classes: int, dino_weights: str | None):
    model = VehicleReIDModel(
        backbone_name=cfg.backbone,
        pretrained=cfg.pretrained,
        embedding_dim=cfg.embedding_dim,
        num_classes=num_classes if cfg.id_weight > 0 else 0,
        pooling=cfg.pooling,
        neck=cfg.neck,
        drop_path_rate=cfg.drop_path_rate,
        grad_checkpointing=cfg.grad_checkpointing,
        arcface=cfg.use_arcface,
        arcface_scale=cfg.arcface_scale,
        arcface_margin=cfg.arcface_margin,
    )
    if dino_weights:
        info = model.load_backbone_weights(dino_weights)
        logger.info("Загружены веса DINO из %s: %d тензоров, пропущено %d",
                    dino_weights, info["loaded"], len(info["skipped"]))
    else:
        logger.info("Стадия 1 не используется — бэкбон остаётся на pretrained-весах")
    if cfg.freeze_blocks:
        model.freeze_blocks(cfg.freeze_blocks)
        logger.info("Заморожены нижние %d блоков бэкбона", cfg.freeze_blocks)
    return model


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    cfg, sources = load_stage_config(args.config, ReIDTrainingConfig)
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.dino_weights:
        cfg.dino_weights = args.dino_weights
    if args.no_dino:
        cfg.dino_weights = None
    cfg.__post_init__()

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_loader, query_loader, gallery_loader, num_classes, split_info = build_loaders(cfg, sources, device)
    model = build_model(cfg, num_classes, cfg.dino_weights)

    trainer = ReIDTrainer(model, cfg, num_classes, device, query_loader, gallery_loader)
    if args.resume:
        path = (ReIDTrainer.find_latest_checkpoint(cfg.checkpoint_dir)
                if args.resume == "auto" else args.resume)
        if path:
            trainer.load_checkpoint(path)
        else:
            logger.warning("Чекпоинт не найден, обучение с нуля")

    # фиксируем сплит — compare_models.py должен оценивать модели на тех же данных
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.checkpoint_dir) / "split_info.json").write_text(
        json.dumps(split_info, indent=2, ensure_ascii=False), encoding="utf-8")

    trainer.train(train_loader)
    logger.info("Лучший mAP %.4f на эпохе %d → %s/best.pt",
                trainer.best_metric, trainer.best_epoch, cfg.checkpoint_dir)


if __name__ == "__main__":
    main()
