"""Метрики retrieval-качества и извлечение эмбеддингов."""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_embeddings(model, dataloader: DataLoader, device, amp_dtype=torch.bfloat16,
                       flip_tta: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Возвращает (embeddings [N, D] L2-нормированные, pids [N], paths)."""
    model.eval()
    embs, pids, paths = [], [], []
    use_amp = device.type == "cuda"
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            feat = model.get_embedding(images, normalize=False)
            if flip_tta:
                feat = feat + model.get_embedding(torch.flip(images, dims=[3]), normalize=False)
        feat = F.normalize(feat.float(), dim=-1)
        embs.append(feat.cpu().numpy().astype(np.float32))
        pids.extend(batch["pid"])
        paths.extend(batch["path"])
    return np.concatenate(embs, axis=0), np.asarray(pids), paths


def compute_metrics(query_emb: np.ndarray, query_pids: np.ndarray,
                    gallery_emb: np.ndarray, gallery_pids: np.ndarray,
                    topk: tuple[int, ...] = (1, 5, 10), max_rank: int = 50,
                    per_query: bool = False) -> dict:
    """mAP, CMC rank-k и mINP (mean Inverse Negative Penalty).

    mINP чувствителен к «самому трудному» позитиву — полезно видеть его рядом с mAP.
    """
    query_pids = np.asarray(query_pids)
    gallery_pids = np.asarray(gallery_pids)
    sim = query_emb @ gallery_emb.T
    order = np.argsort(-sim, axis=1)

    num_q = query_emb.shape[0]
    max_rank = min(max_rank, gallery_emb.shape[0])
    aps, inps, cmc_sum = [], [], np.zeros(max_rank)
    hits1, valid_idx = [], []
    valid_q = 0

    for i in range(num_q):
        matches = (gallery_pids[order[i]] == query_pids[i]).astype(np.int32)
        n_pos = int(matches.sum())
        if n_pos == 0:
            continue
        valid_q += 1

        first = int(np.argmax(matches))
        if first < max_rank:
            cmc_sum[first:] += 1
        hits1.append(float(first == 0))
        valid_idx.append(i)

        cum = np.cumsum(matches)
        precision = cum / (np.arange(len(matches)) + 1)
        aps.append(float((precision * matches).sum() / n_pos))

        last_pos = int(np.nonzero(matches)[0][-1]) + 1  # ранг последнего верного совпадения
        inps.append(n_pos / last_pos)

    if valid_q == 0:
        raise ValueError("Ни у одного query нет позитивов в галерее")

    cmc = cmc_sum / valid_q
    result = {
        "mAP": float(np.mean(aps)),
        "mINP": float(np.mean(inps)),
        "num_query": valid_q,
        "num_gallery": int(gallery_emb.shape[0]),
    }
    for k in topk:
        result[f"rank{k}"] = float(cmc[min(k, max_rank) - 1])
    result["_cmc_curve"] = cmc.tolist()
    if per_query:
        result["_per_query"] = {
            "ap": np.asarray(aps, dtype=np.float64),
            "hit1": np.asarray(hits1, dtype=np.float64),
            "inp": np.asarray(inps, dtype=np.float64),
            "query_index": np.asarray(valid_idx, dtype=np.int64),
        }
    return result


def cmc_curve(metrics: dict) -> np.ndarray:
    return np.asarray(metrics.get("_cmc_curve", []), dtype=np.float32)


@torch.no_grad()
def evaluate_model(model, query_loader: DataLoader, gallery_loader: DataLoader, device,
                   amp_dtype=torch.bfloat16, flip_tta: bool = True) -> dict:
    q_emb, q_pids, _ = extract_embeddings(model, query_loader, device, amp_dtype, flip_tta)
    g_emb, g_pids, _ = extract_embeddings(model, gallery_loader, device, amp_dtype, flip_tta)
    return compute_metrics(q_emb, q_pids, g_emb, g_pids)


# Обратная совместимость со старым API
def compute_map_cmc(query_emb, query_pids, gallery_emb, gallery_pids, topk=(1, 5, 10)):
    metrics = compute_metrics(query_emb, query_pids, gallery_emb, gallery_pids, topk)
    metrics.pop("_cmc_curve", None)
    return metrics
