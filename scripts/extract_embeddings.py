"""Извлечение эмбеддингов для query/gallery обученной моделью.

    python -m scripts.extract_embeddings --config configs/serving.yaml \
        --query-csv data/test_query.csv --gallery-csv data/test_gallery.csv \
        --images-dir data/images
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.vehicle_reid.config import load_serving_config
from src.vehicle_reid.data.datasets import ReIDEvalDataset
from src.vehicle_reid.data.sources import SourceSpec, groups_from_csv
from src.vehicle_reid.data.transforms import build_eval_transforms
from src.vehicle_reid.engine.evaluator import extract_embeddings
from src.vehicle_reid.models.loader import load_reid_model

logger = logging.getLogger("extract")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/serving.yaml")
    p.add_argument("--weights", default=None)
    p.add_argument("--query-csv", required=True)
    p.add_argument("--gallery-csv", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-bbox", action="store_true", help="не резать по bbox (картинки уже кропы)")
    p.add_argument("--require-bbox", action="store_true", help="упасть, если bbox нет в csv")
    return p.parse_args()


def build_loader(csv_path, images_dir, cfg, batch_size, num_workers, device,
                 use_bbox: bool = True, require_bbox: bool = False):
    spec = SourceSpec(name=Path(csv_path).stem, type="csv", csv=csv_path,
                      images_dir=images_dir, group_by="none",
                      use_bbox=use_bbox, require_bbox=require_bbox)
    groups = groups_from_csv(spec)
    ds = ReIDEvalDataset(groups, build_eval_transforms(cfg.image_size), cfg.bbox_padding)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    return loader, ds


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    cfg = load_serving_config(args.config)
    weights = args.weights or cfg.weights
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = load_reid_model(weights, cfg, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_emb, all_ids = [], []
    for name, csv_path in (("query", args.query_csv), ("gallery", args.gallery_csv)):
        loader, ds = build_loader(csv_path, args.images_dir, cfg, args.batch_size,
                                  args.num_workers, device,
                                  use_bbox=not args.no_bbox, require_bbox=args.require_bbox)
        emb, _, paths = extract_embeddings(model, loader, device)
        ids = [Path(p).stem for p in paths]
        np.save(out_dir / f"{name}_embeddings.npy", emb)
        pd.DataFrame({"image_id": ids}).to_csv(out_dir / f"{name}_ids.csv", index=False)
        logger.info("%s: %s -> %s", name, emb.shape, out_dir / f"{name}_embeddings.npy")
        all_emb.append(emb)
        all_ids.extend(ids)

    # совместимость со старым форматом: query, затем gallery в одном файле
    np.save(out_dir / "embeddings.npy", np.concatenate(all_emb, axis=0))
    pd.DataFrame({"image_id": all_ids}).to_csv(out_dir / "embedding_ids.csv", index=False)


if __name__ == "__main__":
    main()
