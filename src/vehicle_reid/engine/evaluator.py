import numpy as np
import torch


@torch.no_grad()
def extract_all_embeddings(model, dataloader, device):
    model.eval()
    embs, ids = [], []
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        emb = model.get_embedding(images)
        embs.append(emb.cpu().numpy())
        ids.extend(batch["image_id"])
    return np.concatenate(embs, axis=0), ids


def compute_map_cmc(query_emb, query_pids, gallery_emb, gallery_pids, topk=(1, 5, 10)):
    """query/gallery_pids — vehicle_id, используется только на валидации (train split)."""
    sim = query_emb @ gallery_emb.T
    order = np.argsort(-sim, axis=1)

    num_q = query_emb.shape[0]
    aps = []
    cmc = np.zeros(max(topk))

    for i in range(num_q):
        matches = (gallery_pids[order[i]] == query_pids[i]).astype(np.int32)
        if matches.sum() == 0:
            continue
        first_match = np.argmax(matches)
        if first_match < len(cmc):
            cmc[first_match:] += 1

        cum_matches = np.cumsum(matches)
        precision_at_k = cum_matches / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / matches.sum()
        aps.append(ap)

    mAP = float(np.mean(aps)) if aps else 0.0
    cmc = cmc / num_q
    result = {"mAP": mAP}
    for k in topk:
        result[f"rank{k}"] = float(cmc[k - 1])
    return result
