"""Единая загрузка обученной ReID-модели для инференса."""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.vehicle_reid.models.model import VehicleReIDModel

logger = logging.getLogger(__name__)


def load_reid_model(weights_path: str | Path, cfg, device) -> VehicleReIDModel:
    """weights_path — чекпоинт стадии 2 (best.pt) либо state_dict бэкбона."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = VehicleReIDModel(
        backbone_name=saved.get("backbone", cfg.backbone),
        pretrained=False,
        embedding_dim=saved.get("embedding_dim", cfg.embedding_dim),
        num_classes=0,                       # классификатор на инференсе не нужен
        pooling=saved.get("pooling", cfg.pooling),
        neck=saved.get("neck", cfg.neck),
        drop_path_rate=0.0,
        grad_checkpointing=False,
    )
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        state = {k: v for k, v in ckpt["model"].items() if not k.startswith("classifier.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("Загрузка весов: missing=%s unexpected=%s", missing, unexpected)
    else:
        info = model.load_backbone_weights(weights_path)
        logger.warning("Чекпоинт без головы: загружено %d тензоров бэкбона, "
                       "BNNeck не обучен", info["loaded"])
    return model.to(device).eval()
