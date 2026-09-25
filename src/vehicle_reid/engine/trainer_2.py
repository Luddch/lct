import logging
import random
from pathlib import Path

import torch
from torch import autocast
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.Services.BatchSizeScheduler import cosine
from src.vehicle_reid.data.transforms import build_global_crop_transforms, build_local_crop_transforms


import copy
import logging
import math
import os
import random
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm




class Trainer:
    def __init__(self, student, config, criterion, optimizer, batch_size_scheduler, device):
        self.cfg = config
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.student = student.to(device)
        # Учитель — EMA-копия студента (включая голову), fp32, без градиентов
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        if hasattr(self.teacher.backbone, "set_grad_checkpointing"):
            self.teacher.backbone.set_grad_checkpointing(False)

        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.bs_scheduler = batch_size_scheduler
        self.use_amp = device.type == "cuda"

        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer: SummaryWriter | None = None

        # Состояние
        self.global_step = 0      # шаги оптимизатора
        self.samples_seen = 0     # машины, прошедшие через оптимизатор
        self.start_epoch = 0
        self.epoch = 0
        self.total_samples = 1
        self._micro_step = 0
        self._accum = 1
        self._loss_sum = torch.zeros((), device=device)
        self._loss_count = 0

    # ------------------------------------------------------------------ train
    def train(self, loader):
        self.total_samples = max(1, self.cfg.epochs * len(loader) * self.cfg.gpu_batch_size)
        self.writer = SummaryWriter(
            log_dir=str(self.ckpt_dir / "tb"),
            purge_step=self.global_step if self.global_step > 0 else None,
        )
        try:
            for epoch in range(self.start_epoch, self.cfg.epochs):
                self.epoch = epoch
                self._train_epoch(loader)
                self.save_checkpoint(next_epoch=epoch + 1)
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self.save_checkpoint(next_epoch=self.epoch)
        except Exception:
            self.logger.exception("Training failed")
            # отдельный файл — не перезаписываем хорошие чекпоинты возможными NaN-весами
            self.save_checkpoint(next_epoch=self.epoch, name="crash.pt")
            raise
        finally:
            self.writer.close()

    def _train_epoch(self, loader):
        self.student.train()
        pbar = tqdm(loader, desc=f"Epoch {self.epoch + 1}/{self.cfg.epochs}", dynamic_ncols=True)
        for batch in pbar:
            if self._micro_step == 0:
                self._accum = self.bs_scheduler.accumulation_steps(self.global_step)
            self._forward_backward(batch)
            self._micro_step += 1
            if self._micro_step == self._accum:
                stats = self._optimizer_step()
                self._micro_step = 0
                pbar.set_postfix(loss=f"{stats['loss']:.4f}", bs=stats["batch_size"],
                                 lr=f"{stats['lr_head']:.2e}", step=self.global_step)

    def _forward_backward(self, batch):
        g = batch["global"].to(self.device, non_blocking=True)   # [B, 2, 3, G, G]
        l = batch["local"].to(self.device, non_blocking=True)    # [B, L, 3, S, S]
        # view-major: [view0 всех машин, view1 всех машин, ...]
        g = g.transpose(0, 1).flatten(0, 1)
        l = l.transpose(0, 1).flatten(0, 1)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            with torch.no_grad():
                t_out = self.teacher(g)          # учитель — только глобальные
            s_out = self.student([g, l])         # студент — все виды

        loss = self.criterion(s_out, t_out, step=self.global_step)
        (loss / self._accum).backward()
        self._loss_sum += loss.detach()
        self._loss_count += 1

    def _optimizer_step(self):
        cfg = self.cfg
        if self.global_step < cfg.backbone_freeze_steps:
            for p in self.student.backbone.parameters():
                p.grad = None
        if self.global_step < cfg.freeze_last_layer_steps:
            self.student.head.last_layer.weight.grad = None

        params = [p for p in self.student.parameters() if p.grad is not None]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.gradient_clip)

        lr_backbone, lr_head = self._set_lr()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        momentum = self._ema_momentum()
        self._update_teacher(momentum)

        batch_size = self._accum * cfg.gpu_batch_size
        self.samples_seen += batch_size
        self.global_step += 1

        loss = (self._loss_sum / max(1, self._loss_count)).item()
        self._loss_sum.zero_()
        self._loss_count = 0
        if not math.isfinite(loss):
            raise FloatingPointError(f"Loss is {loss} at step {self.global_step}")

        stats = {
            "loss": loss,
            "grad_norm": grad_norm.item(),
            "lr_backbone": lr_backbone,
            "lr_head": lr_head,
            "ema_momentum": momentum,
            "batch_size": batch_size,
            "teacher_temp": self.criterion.last_stats.get("teacher_temp", 0.0),
            "teacher_entropy": float(self.criterion.last_stats.get("teacher_entropy", 0.0)),
        }
        if self.global_step % cfg.log_interval == 0:
            self._log(stats)
        if self.global_step % cfg.save_interval == 0:
            self.save_checkpoint(next_epoch=self.epoch)
        return stats

    # ------------------------------------------------------------ schedules
    def _progress(self) -> float:
        return min(1.0, self.samples_seen / self.total_samples)

    def _set_lr(self):
        cfg = self.cfg
        cos = cosine(self._progress(), 1.0, cfg.lr_min_ratio)
        lrs = {True: 0.0, False: 0.0}
        for group in self.optimizer.param_groups:
            # warmup бэкбона начинается после его разморозки
            offset = cfg.backbone_freeze_steps if group["is_backbone"] else 0
            warm = min(1.0, max(0, self.global_step - offset + 1) / max(1, cfg.warmup_steps))
            group["lr"] = group["base_lr"] * warm * cos
            lrs[group["is_backbone"]] = group["lr"]
        return lrs[True], lrs[False]

    def _ema_momentum(self) -> float:
        return cosine(self._progress(), self.cfg.ema_momentum_start, self.cfg.ema_momentum_end)

    @torch.no_grad()
    def _update_teacher(self, m: float):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.lerp_(ps.detach(), 1.0 - m)   # pt = m * pt + (1 - m) * ps
        for bs, bt in zip(self.student.buffers(), self.teacher.buffers()):
            bt.copy_(bs)

    # --------------------------------------------------------------- logging
    def _log(self, stats: dict):
        for k, v in stats.items():
            self.writer.add_scalar(f"train/{k}", v, self.global_step)
        self.logger.info(
            "step %d | loss %.4f | grad_norm %.3f | lr_bb %.2e | lr_head %.2e | bs %d | "
            "ema %.5f | T_teacher %.3f | H_teacher %.3f",
            self.global_step, stats["loss"], stats["grad_norm"], stats["lr_backbone"],
            stats["lr_head"], stats["batch_size"], stats["ema_momentum"],
            stats["teacher_temp"], stats["teacher_entropy"],
        )

    # ----------------------------------------------------------- checkpoints
    def save_checkpoint(self, next_epoch: int, name: str | None = None):
        state = {
            "student": self.student.state_dict(),
            "teacher": self.teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": self.criterion.state_dict(),   # center
            "global_step": self.global_step,
            "samples_seen": self.samples_seen,
            "next_epoch": next_epoch,
            "config": asdict(self.cfg),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        path = self.ckpt_dir / (name or f"step_{self.global_step:07d}.pt")
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)  # атомарно: не получим битый файл при падении во время записи
        self.logger.info("Checkpoint saved: %s (step %d)", path, self.global_step)
        if name is None:
            self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        ckpts = sorted(self.ckpt_dir.glob("step_*.pt"))
        for old in ckpts[: -self.cfg.keep_last_checkpoints]:
            old.unlink(missing_ok=True)

    def load_checkpoint(self, path):
        path = Path(path)
        # map_location="cpu": RNG-состояния должны остаться CPU-тензорами;
        # load_state_dict сам перенесёт веса и состояние оптимизатора на нужный device
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.student.load_state_dict(ckpt["student"])
        self.teacher.load_state_dict(ckpt["teacher"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.criterion.load_state_dict(ckpt["criterion"])
        self.global_step = ckpt["global_step"]
        self.samples_seen = ckpt["samples_seen"]
        self.start_epoch = ckpt["next_epoch"]
        rng = ckpt.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        if ckpt.get("config") != asdict(self.cfg):
            self.logger.warning("Конфиг чекпоинта отличается от текущего")
        self._micro_step = 0
        self.optimizer.zero_grad(set_to_none=True)
        self.logger.info("Resumed from %s: step %d, epoch %d", path, self.global_step, self.start_epoch)

    @staticmethod
    def find_latest_checkpoint(ckpt_dir) -> Path | None:
        ckpts = list(Path(ckpt_dir).glob("step_*.pt"))
        if not ckpts:
            return None
        return max(ckpts, key=lambda p: int(re.findall(r"\d+", p.stem)[0]))

    def export_backbones(self):
        """Для инференса ReID обычно берут бэкбон учителя (он стабильнее)."""
        torch.save(self.teacher.backbone.state_dict(), self.ckpt_dir / "teacher_backbone.pt")
        torch.save(self.student.backbone.state_dict(), self.ckpt_dir / "student_backbone.pt")
        self.logger.info("Backbones exported to %s", self.ckpt_dir)