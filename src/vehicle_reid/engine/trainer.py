import os
import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.vehicle_reid.data.dataset import VehicleReIDDataset
from src.vehicle_reid.data.sampler import PKSampler
from src.vehicle_reid.data.transforms import build_train_transforms, build_eval_transforms
from src.vehicle_reid.models.model import VehicleReIDModel
from src.vehicle_reid.losses.losses import ReIDLossTotal
from src.vehicle_reid.engine.evaluator import extract_all_embeddings, compute_map_cmc
from src.vehicle_reid.utils.seed import set_seed


def split_train_val_by_id(df, val_ratio, seed):
    import numpy as np
    rng = np.random.RandomState(seed)
    ids = df["vehicle_id"].unique()
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return df[df["vehicle_id"].isin(train_ids)].reset_index(drop=True), \
           df[df["vehicle_id"].isin(val_ids)].reset_index(drop=True)


def run_training(cfg):
    set_seed(cfg.project.seed)
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")

    import pandas as pd
    full_df = pd.read_csv(cfg.data.train_csv)
    train_df, val_df = split_train_val_by_id(full_df, cfg.data.val_ratio, cfg.project.seed)

    train_df.to_csv(os.path.join(cfg.project.output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(cfg.project.output_dir, "val_split.csv"), index=False)

    train_tf = build_train_transforms(cfg.data.image_size)
    eval_tf = build_eval_transforms(cfg.data.image_size)

    train_ds = VehicleReIDDataset(
        os.path.join(cfg.project.output_dir, "train_split.csv"),
        cfg.data.images_dir, train_tf, cfg.data.bbox_padding, mode="train")

    # val используем как "query=val, gallery=val" (внутри валидации), pid не пересекаются с train
    val_ds = VehicleReIDDataset(
        os.path.join(cfg.project.output_dir, "val_split.csv"),
        cfg.data.images_dir, eval_tf, cfg.data.bbox_padding, mode="query",
        pid_to_label=None)
    val_ds.mode = "gallery_probe"  # см. ниже — используем image_id + отдельный df с pid

    sampler = PKSampler(train_ds, cfg.train.batch_size_p, cfg.train.batch_size_k)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size_p * cfg.train.batch_size_k,
        sampler=sampler, num_workers=cfg.train.num_workers,
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True)

    model = VehicleReIDModel(
        cfg.model.backbone, cfg.model.pretrained, cfg.model.embedding_dim,
        num_classes=train_ds.num_classes, pooling=cfg.model.pooling,
        arcface_scale=cfg.loss.arcface_scale, arcface_margin=cfg.loss.arcface_margin,
    ).to(device)

    criterion = ReIDLossTotal(cfg.loss, train_ds.num_classes, cfg.model.embedding_dim, device)

    params = list(model.parameters())
    if criterion.center_criterion is not None:
        params += list(criterion.center_criterion.parameters())

    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    def lr_lambda(epoch):
        if epoch < cfg.train.warmup_epochs:
            return (epoch + 1) / cfg.train.warmup_epochs
        progress = (epoch - cfg.train.warmup_epochs) / max(1, cfg.train.epochs - cfg.train.warmup_epochs)
        import math
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp)

    best_map = -1.0
    patience = 0
    best_state = None

    for epoch in range(cfg.train.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.train.amp):
                outputs = model(images, labels)
                loss, logs = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(logs)

        scheduler.step()

        if (epoch + 1) % cfg.train.eval_every == 0 or (epoch + 1) == cfg.train.epochs:
            metrics = evaluate_val(model, val_df, cfg, device)
            print(f"[Val] epoch={epoch+1} mAP={metrics['mAP']:.4f} rank1={metrics['rank1']:.4f}")

            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
                os.makedirs(cfg.project.weights_dir, exist_ok=True)
                torch.save(best_state, os.path.join(cfg.project.weights_dir, "best_model.pth"))
            else:
                patience += 1
                if patience >= cfg.train.early_stop_patience:
                    print("Early stopping.")
                    break

    return model


@torch.no_grad()
def evaluate_val(model, val_df, cfg, device):
    """Half-half split val по vehicle_id: первая съёмка -> query, остальные -> gallery."""
    import numpy as np
    from src.vehicle_reid.data.transforms import build_eval_transforms
    from src.vehicle_reid.data.dataset import VehicleReIDDataset

    val_df = val_df.copy()
    val_df["_rank"] = val_df.groupby("vehicle_id").cumcount()
    query_df = val_df[val_df["_rank"] == 0].drop(columns=["_rank"])
    gallery_df = val_df[val_df["_rank"] > 0].drop(columns=["_rank"])

    if len(gallery_df) == 0 or len(query_df) == 0:
        return {"mAP": 0.0, "rank1": 0.0, "rank5": 0.0, "rank10": 0.0}

    tmp_dir = cfg.project.output_dir
    q_path = os.path.join(tmp_dir, "_val_query.csv")
    g_path = os.path.join(tmp_dir, "_val_gallery.csv")
    query_df.to_csv(q_path, index=False)
    gallery_df.to_csv(g_path, index=False)

    eval_tf = build_eval_transforms(cfg.data.image_size)
    q_ds = VehicleReIDDataset(q_path, cfg.data.images_dir, eval_tf, cfg.data.bbox_padding, mode="gallery")
    g_ds = VehicleReIDDataset(g_path, cfg.data.images_dir, eval_tf, cfg.data.bbox_padding, mode="gallery")

    from torch.utils.data import DataLoader
    q_loader = DataLoader(q_ds, batch_size=128, num_workers=2)
    g_loader = DataLoader(g_ds, batch_size=128, num_workers=2)

    q_emb, q_ids = extract_all_embeddings(model, q_loader, device)
    g_emb, g_ids = extract_all_embeddings(model, g_loader, device)

    id_map_q = query_df.set_index("image_id")["vehicle_id"].to_dict()
    id_map_g = gallery_df.set_index("image_id")["vehicle_id"].to_dict()
    q_pids = np.array([id_map_q[i] for i in q_ids])
    g_pids = np.array([id_map_g[i] for i in g_ids])

    from src.vehicle_reid.engine.evaluator import compute_map_cmc
    return compute_map_cmc(q_emb, q_pids, g_emb, g_pids)
