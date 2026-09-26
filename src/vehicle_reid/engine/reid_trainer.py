"""Стадия 2: contrastive (SupCon с memory-очередью) + ID-loss.

Экономия VRAM:
  * micro-batch + градиентная аккумуляция (эффективный батч не влияет на память);
  * gradient checkpointing бэкбона;
  * заморозка нижних блоков ViT;
  * negatives берутся из очереди, а не из батча — поэтому batch-hard triplet не нужен;
  * momentum-энкодер работает в no_grad, его активации не хранятся.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.Services.BatchSizeScheduler import cosine
from src.vehicle_reid.engine.evaluator import evaluate_model
from src.vehicle_reid.engine.reid_config import ReIDTrainingConfig
from src.vehicle_reid.losses.losses import CenterLoss, CrossEntropyLabelSmooth, SupConMemoryLoss

logger = logging.getLogger(__name__)

_AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


class ReIDTrainer:
    def __init__(self, model, cfg: ReIDTrainingConfig, num_classes: int, device,
                 query_loader=None, gallery_loader=None):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.num_classes = num_classes
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader

        self.amp_dtype = _AMP_DTYPES.get(cfg.amp_dtype)
        self.use_amp = self.amp_dtype is not None and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and self.amp_dtype is torch.float16)

        dim = model.embedding_dim
        self.supcon = SupConMemoryLoss(dim=dim, queue_size=cfg.queue_size,
                                       temperature=cfg.supcon_temperature).to(device)
        self.id_loss = CrossEntropyLabelSmooth(cfg.label_smoothing).to(device)
        self.center_loss = (CenterLoss(num_classes, dim).to(device)
                            if cfg.center_weight > 0 else None)

        self.momentum_model = None
        if cfg.use_momentum_encoder:
            self.momentum_model = copy.deepcopy(self.model).to(device)
            self.momentum_model.requires_grad_(False)
            self.momentum_model.eval()
            if hasattr(self.momentum_model.backbone, "set_grad_checkpointing"):
                self.momentum_model.backbone.set_grad_checkpointing(False)

        self.optimizer = self._build_optimizer()

        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer: SummaryWriter | None = None

        self.global_step = 0
        self.epoch = 0
        self.start_epoch = 0
        self.total_steps = 1
        self.best_metric = -1.0
        self.best_epoch = -1
        self.epochs_without_improve = 0
        self.history: list[dict] = []

    # ---------------------------------------------------------------- setup
    def _build_optimizer(self):
        cfg = self.cfg
        skip = set()
        if hasattr(self.model.backbone, "no_weight_decay"):
            skip = {f"backbone.{n}" for n in self.model.backbone.no_weight_decay()}

        buckets: dict[tuple[bool, bool], list] = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_backbone = name.startswith("backbone.")
            no_decay = p.ndim <= 1 or name.endswith(".bias") or name in skip
            buckets.setdefault((is_backbone, not no_decay), []).append(p)
        if self.center_loss is not None:
            buckets.setdefault((False, False), []).extend(self.center_loss.parameters())

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
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info("Обучаемых параметров: %.1fM из %.1fM", n_train / 1e6, n_total / 1e6)
        return torch.optim.AdamW(groups, betas=(0.9, 0.999),
                                 fused=self.device.type == "cuda")

    # ---------------------------------------------------------------- train
    def train(self, loader):
        cfg = self.cfg
        steps_per_epoch = max(1, len(loader) // cfg.accumulation_steps)
        self.total_steps = max(1, cfg.epochs * steps_per_epoch)
        self.writer = SummaryWriter(log_dir=str(self.ckpt_dir / "tb"))
        logger.info("Эффективный батч: %d изображений (%d x %d x %d видов)",
                    cfg.effective_batch, cfg.micro_batch_size, cfg.accumulation_steps, cfg.n_views)
        try:
            for epoch in range(self.start_epoch, cfg.epochs):
                self.epoch = epoch
                self._train_epoch(loader)
                if self._maybe_evaluate():
                    break
            self._save_history()
        except KeyboardInterrupt:
            logger.warning("Обучение прервано пользователем")
            self.save_checkpoint("interrupted.pt")
            self._save_history()
        finally:
            if self.writer:
                self.writer.close()
        return self.history

    def _train_epoch(self, loader):
        cfg = self.cfg
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"Epoch {self.epoch + 1}/{cfg.epochs}", dynamic_ncols=True)
        accum_stats: dict[str, float] = {}

        for micro_step, batch in enumerate(pbar):
            stats = self._forward_backward(batch)
            for k, v in stats.items():
                accum_stats[k] = accum_stats.get(k, 0.0) + v

            if (micro_step + 1) % cfg.accumulation_steps == 0:
                grad_norm = self._optimizer_step()
                n = cfg.accumulation_steps
                logged = {k: v / n for k, v in accum_stats.items()}
                logged["grad_norm"] = grad_norm
                accum_stats.clear()
                pbar.set_postfix(loss=f"{logged['loss']:.3f}",
                                 supcon=f"{logged['supcon']:.3f}",
                                 id=f"{logged['id']:.3f}",
                                 step=self.global_step)
                if self.global_step % cfg.log_interval == 0:
                    self._log(logged)

    def _forward_backward(self, batch) -> dict[str, float]:
        cfg = self.cfg
        views = batch["views"].to(self.device, non_blocking=True)   # [B, V, 3, H, W]
        labels = batch["label"].to(self.device, non_blocking=True)  # [B]
        b, v = views.shape[:2]
        images = views.flatten(0, 1)                                # [B*V, ...]
        flat_labels = labels.repeat_interleave(v)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.float32,
                            enabled=self.use_amp):
            out = self.model(images, flat_labels)

        feat_t = out["feat_t"].float()
        loss = torch.zeros((), device=self.device)
        stats: dict[str, float] = {}

        # --- ключи для очереди
        keys, key_labels = None, None
        if self.momentum_model is not None:
            with torch.no_grad(), torch.autocast(device_type=self.device.type,
                                                 dtype=self.amp_dtype or torch.float32,
                                                 enabled=self.use_amp):
                keys = self.momentum_model(images)["feat_t"].float()
            key_labels = flat_labels

        supcon = self.supcon(feat_t, flat_labels, keys=keys, key_labels=key_labels)
        loss = loss + cfg.supcon_weight * supcon
        stats["supcon"] = float(supcon.detach())

        if out["logits"] is not None and cfg.id_weight > 0:
            id_loss = self.id_loss(out["logits"], flat_labels)
            loss = loss + cfg.id_weight * id_loss
            stats["id"] = float(id_loss.detach())
            with torch.no_grad():
                stats["acc"] = float((out["logits"].argmax(1) == flat_labels).float().mean())
        else:
            stats["id"] = 0.0

        if self.center_loss is not None:
            c = self.center_loss(feat_t, flat_labels)
            loss = loss + cfg.center_weight * c
            stats["center"] = float(c.detach())

        stats["loss"] = float(loss.detach())
        if not math.isfinite(stats["loss"]):
            raise FloatingPointError(f"Loss = {stats['loss']} на шаге {self.global_step}")

        self.scaler.scale(loss / cfg.accumulation_steps).backward()
        return stats

    def _optimizer_step(self) -> float:
        cfg = self.cfg
        if self.global_step < cfg.freeze_backbone_steps:
            for name, p in self.model.named_parameters():
                if name.startswith("backbone."):
                    p.grad = None

        self.scaler.unscale_(self.optimizer)
        params = [p for p in self.model.parameters() if p.grad is not None]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.gradient_clip)

        self._set_lr()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if self.momentum_model is not None:
            self._update_momentum(self._momentum_value())
        self.global_step += 1
        return float(grad_norm)

    def _set_lr(self):
        cfg = self.cfg
        progress = min(1.0, self.global_step / self.total_steps)
        cos = cosine(progress, 1.0, cfg.lr_min_ratio)
        for group in self.optimizer.param_groups:
            offset = cfg.freeze_backbone_steps if group["is_backbone"] else 0
            warm = min(1.0, max(0, self.global_step - offset + 1) / max(1, cfg.warmup_steps))
            group["lr"] = group["base_lr"] * warm * cos

    def _momentum_value(self) -> float:
        progress = min(1.0, self.global_step / self.total_steps)
        return cosine(progress, self.cfg.momentum_start, self.cfg.momentum_end)

    @torch.no_grad()
    def _update_momentum(self, m: float):
        for ps, pm in zip(self.model.parameters(), self.momentum_model.parameters()):
            pm.lerp_(ps.detach(), 1.0 - m)
        for bs, bm in zip(self.model.buffers(), self.momentum_model.buffers()):
            bm.copy_(bs)

    # ----------------------------------------------------------- валидация
    def _maybe_evaluate(self) -> bool:
        """Возвращает True, если сработал early stopping."""
        cfg = self.cfg
        is_last = self.epoch == cfg.epochs - 1
        if self.query_loader is None or ((self.epoch + 1) % cfg.eval_every and not is_last):
            self.save_checkpoint(f"epoch_{self.epoch + 1:03d}.pt")
            return False

        metrics = evaluate_model(self.model, self.query_loader, self.gallery_loader,
                                 self.device, self.amp_dtype or torch.float32)
        metrics.pop("_cmc_curve", None)
        metrics["epoch"] = self.epoch + 1
        self.history.append(metrics)
        logger.info("Валидация эпоха %d: mAP %.4f | rank1 %.4f | rank5 %.4f | mINP %.4f",
                    self.epoch + 1, metrics["mAP"], metrics["rank1"], metrics["rank5"], metrics["mINP"])
        if self.writer:
            for k, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(f"val/{k}", val, self.global_step)

        score = metrics["mAP"]
        self.save_checkpoint(f"epoch_{self.epoch + 1:03d}.pt")
        if score > self.best_metric:
            self.best_metric = score
            self.best_epoch = self.epoch + 1
            self.epochs_without_improve = 0
            self.save_checkpoint("best.pt", metrics=metrics)
            logger.info("Новый лучший чекпоинт: mAP %.4f", score)
        else:
            self.epochs_without_improve += 1
            if self.epochs_without_improve >= self.cfg.early_stop_patience:
                logger.info("Early stopping: %d эпох без улучшения", self.epochs_without_improve)
                return True
        return False

    # --------------------------------------------------------- чекпоинты
    def save_checkpoint(self, name: str, metrics: dict | None = None):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "supcon": self.supcon.state_dict(),
            "global_step": self.global_step,
            "next_epoch": self.epoch + 1,
            "best_metric": self.best_metric,
            "num_classes": self.num_classes,
            "config": asdict(self.cfg),
            "metrics": metrics,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }
        path = self.ckpt_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
        if name.startswith("epoch_"):
            self._rotate()

    def _rotate(self):
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        for old in ckpts[: -self.cfg.keep_last_checkpoints]:
            old.unlink(missing_ok=True)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.supcon.load_state_dict(ckpt["supcon"])
        self.global_step = ckpt["global_step"]
        self.start_epoch = ckpt["next_epoch"]
        self.best_metric = ckpt.get("best_metric", -1.0)
        if self.momentum_model is not None:
            self.momentum_model.load_state_dict(ckpt["model"])
        logger.info("Возобновление с %s (шаг %d, эпоха %d)", path, self.global_step, self.start_epoch)

    @staticmethod
    def find_latest_checkpoint(ckpt_dir) -> Path | None:
        ckpts = list(Path(ckpt_dir).glob("epoch_*.pt"))
        if not ckpts:
            return None
        return max(ckpts, key=lambda p: int(re.findall(r"\d+", p.stem)[0]))

    def _save_history(self):
        if not self.history:
            return
        path = self.ckpt_dir / "val_history.json"
        path.write_text(json.dumps(
            {"best_epoch": self.best_epoch, "best_mAP": self.best_metric, "history": self.history},
            indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("История валидации: %s", path)

    def _log(self, stats: dict):
        if self.writer:
            for k, v in stats.items():
                self.writer.add_scalar(f"train/{k}", v, self.global_step)
            self.writer.add_scalar("train/lr_head", self.optimizer.param_groups[0]["lr"], self.global_step)
