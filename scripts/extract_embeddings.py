import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.vehicle_reid.config import load_config
from src.vehicle_reid.data.dataset import VehicleReIDDataset
from src.vehicle_reid.data.transforms import build_eval_transforms
from src.vehicle_reid.models.model import VehicleReIDModel


def load_model(cfg, weights_path):
    # num_classes здесь не важен — arc_head не используется при инференсе
    # и будет исключён из загрузки ниже, если его форма не совпадает
    model = VehicleReIDModel(
        cfg.model.backbone, pretrained=False, embedding_dim=cfg.model.embedding_dim,
        num_classes=1, pooling=cfg.model.pooling,
    )

    checkpoint_state = torch.load(weights_path, map_location="cpu")
    model_state = model.state_dict()

    # Оставляем только те тензоры, чья форма совпадает с текущей моделью.
    # arc_head.weight (и другие head-специфичные слои) будут отброшены,
    # т.к. для инференса они не нужны — используется только backbone+bottleneck
    filtered_state = {
        k: v for k, v in checkpoint_state.items()
        if k in model_state and v.shape == model_state[k].shape
    }

    skipped = sorted(set(checkpoint_state.keys()) - set(filtered_state.keys()))
    if skipped:
        print(f"[load_model] Пропущены несовместимые/неиспользуемые ключи: {skipped}")

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"[load_model] Missing keys (не критично для inference): {missing}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--weights", default="weights/best_model.pth")
    parser.add_argument("--out", default="outputs/embeddings.npy")
    parser.add_argument("--ids_out", default="outputs/embedding_ids.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")

    model = load_model(cfg, args.weights).to(device).eval()
    eval_tf = build_eval_transforms(cfg.data.image_size)

    query_ds = VehicleReIDDataset(cfg.data.query_csv, cfg.data.images_dir, eval_tf,
                                   cfg.data.bbox_padding, mode="query")
    gallery_ds = VehicleReIDDataset(cfg.data.gallery_csv, cfg.data.images_dir, eval_tf,
                                     cfg.data.bbox_padding, mode="gallery")

    query_loader = DataLoader(query_ds, batch_size=128, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_ds, batch_size=128, shuffle=False, num_workers=4)

    all_embs, all_ids = [], []
    with torch.no_grad():
        for loader in (query_loader, gallery_loader):
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                emb = model.get_embedding(images).cpu().numpy().astype(np.float32)
                all_embs.append(emb)
                all_ids.extend(batch["image_id"])

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, embeddings)
    pd.DataFrame({"image_id": all_ids}).to_csv(args.ids_out, index=False)
    print(f"Saved embeddings: {embeddings.shape} -> {args.out}")


if __name__ == "__main__":
    main()
