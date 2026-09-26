"""Стадия 1: DINO self-distillation на нескольких (не)размеченных датасетах.

    python -m scripts.train_dino --config configs/dino.yaml
"""
from __future__ import annotations

import argparse
import logging
import random

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader

from src.Services.BatchSizeScheduler import BatchSizeScheduler
from src.vehicle_reid.config import load_stage_config
from src.vehicle_reid.data.datasets import MultiCropDataset
from src.vehicle_reid.data.sources import build_groups, describe
from src.vehicle_reid.data.transforms import build_global_crop_transforms, build_local_crop_transforms
from src.vehicle_reid.engine.train_config import TrainingConfig
from src.vehicle_reid.engine.trainer_2 import Trainer
from src.vehicle_reid.losses.losses import DINOLoss
from src.vehicle_reid.models.model import DINOHead, MultiCropWrapper


def parse_args():
    p = argparse.ArgumentParser("DINO pretraining for vehicle ReID")
    p.add_argument("--config", default="configs/dino.yaml")
    p.add_argument("--resume", default=None, help="путь к чекпоинту или 'auto'")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--gpu-batch-size", type=int, default=None)
    p.add_argument("--checkpoint-dir", default=None)
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_model(cfg: TrainingConfig) -> MultiCropWrapper:
    backbone = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=0,
        dynamic_img_size=True,     # нужно для кропов 224 и 96 в одной модели
        drop_path_rate=cfg.drop_path_rate,
    )
    if cfg.grad_checkpointing:
        backbone.set_grad_checkpointing(True)
    if cfg.freeze_blocks:
        for name in ("patch_embed",):
            module = getattr(backbone, name, None)
            if module is not None:
                module.requires_grad_(False)
        for blk in list(getattr(backbone, "blocks", []))[: cfg.freeze_blocks]:
            blk.requires_grad_(False)
    head = DINOHead(
        in_dim=backbone.num_features,
        out_dim=cfg.out_dim,
        hidden_dim=cfg.head_hidden_dim,
        bottleneck_dim=cfg.head_bottleneck_dim,
    )
    return MultiCropWrapper(backbone, head)  # веса в fp32! bf16 — только через autocast


def build_param_groups(model: MultiCropWrapper, cfg: TrainingConfig):
    skip = set()
    if hasattr(model.backbone, "no_weight_decay"):
        skip = {f"backbone.{n}" for n in model.backbone.no_weight_decay()}

    buckets: dict[tuple[bool, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        no_decay = p.ndim <= 1 or name.endswith(".bias") or name in skip
        buckets.setdefault((is_backbone, not no_decay), []).append(p)

    groups = []
    for (is_backbone, decay), params in buckets.items():
        base_lr = cfg.lr_backbone if is_backbone else cfg.lr_head
        groups.append({
            "params": params,
            "weight_decay": cfg.weight_decay if decay else 0.0,
            "lr": base_lr,
            "base_lr": base_lr,
            "is_backbone": is_backbone,
        })
    return groups


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("train_dino")
    args = parse_args()

    cfg, sources = load_stage_config(args.config, TrainingConfig)
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.gpu_batch_size:
        cfg.gpu_batch_size = args.gpu_batch_size
    cfg.__post_init__()

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- данные: несколько источников сразу
    groups = build_groups(sources, require_pid=False)
    stats = describe(groups)
    logger.info("Источники: %s", stats.per_source)

    dataset = MultiCropDataset(
        groups=groups,
        global_transforms=build_global_crop_transforms(cfg.global_size),
        local_transform=build_local_crop_transforms(cfg.local_size),
        n_local_crops=cfg.n_local_crops,
        bbox_padding=cfg.bbox_padding,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.gpu_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        worker_init_fn=seed_worker,
    )
    if len(loader) == 0:
        raise ValueError("Групп меньше, чем gpu_batch_size")
    logger.info("Групп: %d, micro-batches на эпоху: %d", len(dataset), len(loader))

    # ---- модель / оптимизация
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg), betas=(0.9, 0.999), fused=device.type == "cuda"
    )
    criterion = DINOLoss(
        out_dim=cfg.out_dim,
        n_global_crops=2,
        n_local_crops=cfg.n_local_crops,
        student_temp=cfg.student_temp,
        teacher_temp=cfg.teacher_temp,
        warmup_teacher_temp=cfg.warmup_teacher_temp,
        teacher_temp_warmup_steps=cfg.teacher_temp_warmup_steps,
        center_momentum=cfg.center_momentum,
    )
    bs_scheduler = BatchSizeScheduler(
        start=cfg.batch_size_start,
        end=cfg.batch_size_end,
        warmup_steps=cfg.batch_size_warmup_steps,
        micro_batch_size=cfg.gpu_batch_size,
    )

    trainer = Trainer(model, cfg, criterion, optimizer, bs_scheduler, device)

    if args.resume:
        path = Trainer.find_latest_checkpoint(cfg.checkpoint_dir) if args.resume == "auto" else args.resume
        if path:
            trainer.load_checkpoint(path)
        else:
            logger.warning("Чекпоинт не найден, обучение с нуля")

    trainer.train(loader)
    trainer.export_backbones()
    logger.info("Готово. Веса для стадии 2: %s/teacher_backbone.pt", cfg.checkpoint_dir)


if __name__ == "__main__":
    main()
