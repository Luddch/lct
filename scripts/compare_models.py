"""Сравнение моделей ReID на одном и том же валидационном сплите.

Оценивает произвольное число моделей (базовая pretrained, после DINO,
после contrastive fine-tuning) и печатает таблицу mAP / Rank-k / mINP с дельтами
относительно baseline, плюс парный бутстрэп для проверки, что улучшение не шум.

    python -m scripts.compare_models --config configs/reid.yaml \
        --model "baseline=pretrained" \
        --model "dino=checkpoints/dino/teacher_backbone.pt" \
        --model "finetuned=checkpoints/reid/best.pt" \
        --out outputs/comparison

Формат --model: ИМЯ=СПЕЦ, где СПЕЦ:
    pretrained                       — бэкбон из timm «как есть» (нулевая точка отсчёта);
    путь к teacher_backbone.pt       — веса бэкбона после стадии 1;
    путь к best.pt (стадия 2)        — полный чекпоинт ReID-модели.
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
from src.vehicle_reid.data.datasets import ReIDEvalDataset, make_query_gallery, split_groups_by_id
from src.vehicle_reid.data.sources import build_groups
from src.vehicle_reid.data.transforms import build_eval_transforms
from src.vehicle_reid.engine.evaluator import compute_metrics, extract_embeddings
from src.vehicle_reid.engine.reid_config import ReIDTrainingConfig
from src.vehicle_reid.models.model import VehicleReIDModel

logger = logging.getLogger("compare")
METRIC_KEYS = ("mAP", "rank1", "rank5", "rank10", "mINP")


def parse_args():
    p = argparse.ArgumentParser("Сравнение базовой и дообученной моделей")
    p.add_argument("--config", default="configs/reid.yaml")
    p.add_argument("--model", action="append", required=True, help="ИМЯ=СПЕЦ, можно повторять")
    p.add_argument("--out", default="outputs/comparison")
    p.add_argument("--baseline", default=None, help="имя модели-эталона (по умолчанию первая)")
    p.add_argument("--bootstrap", type=int, default=1000, help="итераций парного бутстрэпа, 0 = выкл")
    p.add_argument("--qualitative", type=int, default=6, help="сколько query показать на картинках")
    p.add_argument("--topk-vis", type=int, default=5)
    p.add_argument("--no-flip-tta", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


# ------------------------------------------------------------------- модели


def load_model(spec: str, cfg: ReIDTrainingConfig, device):
    """Собирает модель по спецификации. Инференс использует только backbone+BNNeck."""
    if spec == "pretrained":
        model = VehicleReIDModel(cfg.backbone, pretrained=True, embedding_dim=cfg.embedding_dim,
                                 num_classes=0, pooling=cfg.pooling, neck=cfg.neck,
                                 drop_path_rate=0.0, grad_checkpointing=False)
        return model.to(device).eval()

    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"Нет файла {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    is_reid_ckpt = isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict)
    if is_reid_ckpt:
        saved = ckpt.get("config", {})
        model = VehicleReIDModel(
            saved.get("backbone", cfg.backbone), pretrained=False,
            embedding_dim=saved.get("embedding_dim", cfg.embedding_dim),
            num_classes=ckpt.get("num_classes", 0) if saved.get("id_weight", 1) > 0 else 0,
            pooling=saved.get("pooling", cfg.pooling), neck=saved.get("neck", cfg.neck),
            drop_path_rate=0.0, grad_checkpointing=False,
            arcface=saved.get("use_arcface", True),
        )
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            logger.warning("%s: missing=%d unexpected=%d", path.name, len(missing), len(unexpected))
        logger.info("%s: чекпоинт стадии 2, метрики при сохранении: %s", path.name, ckpt.get("metrics"))
    else:
        # просто веса бэкбона (стадия 1): BNNeck не обучен — это честная «DINO-only» точка
        model = VehicleReIDModel(cfg.backbone, pretrained=False, embedding_dim=cfg.embedding_dim,
                                 num_classes=0, pooling=cfg.pooling, neck=cfg.neck,
                                 drop_path_rate=0.0, grad_checkpointing=False)
        info = model.load_backbone_weights(path)
        logger.info("%s: загружено %d тензоров бэкбона", path.name, info["loaded"])
    return model.to(device).eval()


# ------------------------------------------------------------------ статистика


def paired_bootstrap(base: np.ndarray, other: np.ndarray, n_iter: int = 1000, seed: int = 0):
    """Доверительный интервал разницы метрики по одним и тем же query."""
    if n_iter <= 0 or len(base) == 0 or len(base) != len(other):
        return None
    rng = np.random.default_rng(seed)
    n = len(base)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        diffs[i] = other[idx].mean() - base[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "delta": float(other.mean() - base.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_improved": float((diffs > 0).mean()),   # доля бутстрэп-выборок с улучшением
    }


# ------------------------------------------------------------------- отчёты


def markdown_table(results: dict[str, dict], baseline: str) -> str:
    header = "| Модель | " + " | ".join(METRIC_KEYS) + " |"
    sep = "|---" * (len(METRIC_KEYS) + 1) + "|"
    lines = [header, sep]
    base = results[baseline]
    for name, m in results.items():
        cells = []
        for k in METRIC_KEYS:
            if name == baseline:
                cells.append(f"{m[k]:.4f}")
            else:
                d = m[k] - base[k]
                rel = (d / base[k] * 100) if base[k] > 0 else float("nan")
                cells.append(f"{m[k]:.4f} ({d:+.4f}, {rel:+.1f}%)")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_cmc(results: dict[str, dict], curves: dict[str, np.ndarray], out_path: Path, max_rank: int = 20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, curve in curves.items():
        r = min(max_rank, len(curve))
        ax.plot(range(1, r + 1), curve[:r], marker="o", markersize=3,
                label=f"{name} (mAP {results[name]['mAP']:.3f})")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative Matching Characteristic")
    ax.set_title("CMC-кривые на валидационном сплите")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_metric_bars(results: dict[str, dict], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    x = np.arange(len(METRIC_KEYS))
    width = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(names):
        vals = [results[name][k] for k in METRIC_KEYS]
        bars = ax.bar(x + i * width, vals, width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(METRIC_KEYS)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Сравнение метрик")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_retrieval(embeddings: dict[str, tuple], q_paths, g_paths, out_dir: Path,
                   n_query: int, topk: int, seed: int = 0):
    """Качественное сравнение: для одних и тех же query рисуем top-k каждой модели."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.vehicle_reid.data.datasets import load_crop
    from src.vehicle_reid.data.sources import Record

    rng = random.Random(seed)
    first = next(iter(embeddings.values()))
    n_q = first[0].shape[0]
    picks = rng.sample(range(n_q), min(n_query, n_q))

    def show(ax, path, title, color=None):
        img = load_crop(Record(path=path))
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=7, color=color or "black")
        if color:
            for s in ax.spines.values():
                s.set_color(color); s.set_linewidth(2.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    for qi in picks:
        n_rows = len(embeddings)
        fig, axes = plt.subplots(n_rows, topk + 1, figsize=(1.6 * (topk + 1), 1.9 * n_rows),
                                 squeeze=False)
        for row, (name, (q_emb, q_pids, g_emb, g_pids)) in enumerate(embeddings.items()):
            sims = g_emb @ q_emb[qi]
            order = np.argsort(-sims)[:topk]
            show(axes[row][0], q_paths[qi], f"{name}\nquery {q_pids[qi]}")
            for col, gi in enumerate(order, start=1):
                ok = g_pids[gi] == q_pids[qi]
                show(axes[row][col], g_paths[gi], f"{sims[gi]:.3f}",
                     color="green" if ok else "red")
        fig.tight_layout()
        fig.savefig(out_dir / f"query_{qi:04d}.png", dpi=130)
        plt.close(fig)
    logger.info("Примеры выдачи: %s (%d файлов)", out_dir, len(picks))


# --------------------------------------------------------------------- main


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    cfg, sources = load_stage_config(args.config, ReIDTrainingConfig)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Тот же сплит, что и в обучении: детерминируется cfg.seed / cfg.val_ratio
    groups = build_groups(sources, require_pid=True)
    _, val_groups = split_groups_by_id(groups, cfg.val_ratio, cfg.seed)
    query_groups, gallery_groups = make_query_gallery(val_groups, cfg.seed, cfg.n_query_per_id)

    eval_tf = build_eval_transforms(cfg.image_size)
    query_ds = ReIDEvalDataset(query_groups, eval_tf, cfg.bbox_padding)
    gallery_ds = ReIDEvalDataset(gallery_groups, eval_tf, cfg.bbox_padding)
    pin = device.type == "cuda"
    query_loader = DataLoader(query_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=pin)
    gallery_loader = DataLoader(gallery_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=pin)
    logger.info("Валидация: %d query, %d gallery, %d ID",
                len(query_ds), len(gallery_ds), len(query_groups))

    specs = []
    for item in args.model:
        if "=" not in item:
            raise ValueError(f"--model ожидает формат ИМЯ=СПЕЦ, получено: {item}")
        name, spec = item.split("=", 1)
        specs.append((name.strip(), spec.strip()))
    baseline = args.baseline or specs[0][0]

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    results, curves, per_query, embeddings = {}, {}, {}, {}
    q_paths = g_paths = None

    for name, spec in specs:
        logger.info("=== %s (%s)", name, spec)
        model = load_model(spec, cfg, device)
        q_emb, q_pids, q_paths = extract_embeddings(model, query_loader, device, amp_dtype,
                                                    flip_tta=not args.no_flip_tta)
        g_emb, g_pids, g_paths = extract_embeddings(model, gallery_loader, device, amp_dtype,
                                                    flip_tta=not args.no_flip_tta)
        m = compute_metrics(q_emb, q_pids, g_emb, g_pids, per_query=True)
        curves[name] = np.asarray(m.pop("_cmc_curve"))
        per_query[name] = m.pop("_per_query")
        results[name] = m
        embeddings[name] = (q_emb, q_pids, g_emb, g_pids)
        logger.info("%s: mAP %.4f | rank1 %.4f | rank5 %.4f | mINP %.4f",
                    name, m["mAP"], m["rank1"], m["rank5"], m["mINP"])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- значимость улучшений
    significance = {}
    base_pq = per_query[baseline]
    for name, pq in per_query.items():
        if name == baseline:
            continue
        significance[name] = {
            "mAP": paired_bootstrap(base_pq["ap"], pq["ap"], args.bootstrap),
            "rank1": paired_bootstrap(base_pq["hit1"], pq["hit1"], args.bootstrap),
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = markdown_table(results, baseline)

    report = [
        "# Сравнение моделей ReID", "",
        f"- валидационный сплит: {len(query_ds)} query / {len(gallery_ds)} gallery, "
        f"{len(query_groups)} ID (disjoint от train, seed={cfg.seed}, val_ratio={cfg.val_ratio})",
        f"- baseline: **{baseline}**", f"- flip-TTA: {not args.no_flip_tta}", "",
        table, "",
    ]
    if significance:
        report += ["## Парный бутстрэп (95% ДИ разницы с baseline)", "",
                   "| Модель | метрика | Δ | CI95 | доля бутстрэпов с улучшением |",
                   "|---|---|---|---|---|"]
        for name, metrics in significance.items():
            for metric, s in metrics.items():
                if s is None:
                    continue
                report.append(f"| {name} | {metric} | {s['delta']:+.4f} | "
                              f"[{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}] | {s['p_improved']:.3f} |")
        report.append("")
        report.append("Улучшение считается уверенным, если нижняя граница CI95 > 0.")

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(
        {"baseline": baseline, "results": results, "significance": significance,
         "cmc": {k: v.tolist() for k, v in curves.items()}},
        indent=2, ensure_ascii=False), encoding="utf-8")

    plot_cmc(results, curves, out_dir / "cmc.png")
    plot_metric_bars(results, out_dir / "metrics.png")
    if args.qualitative > 0:
        plot_retrieval(embeddings, q_paths, g_paths, out_dir / "retrieval",
                       args.qualitative, args.topk_vis, cfg.seed)

    print("\n" + table + "\n")
    print(f"Отчёт: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
