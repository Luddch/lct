import argparse
import numpy as np
import pandas as pd

from src.vehicle_reid.config import load_config
from src.vehicle_reid.search.index import VectorIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--embeddings", default="outputs/embeddings.npy")
    parser.add_argument("--ids", default="outputs/embedding_ids.csv")
    parser.add_argument("--query_csv", default="dataset/test_query.csv")
    parser.add_argument("--gallery_csv", default="dataset/test_gallery.csv")
    parser.add_argument("--submission_out", default="outputs/submission.csv")
    parser.add_argument("--candidates_out", default="outputs/candidates.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)

    embeddings = np.load(args.embeddings)
    ids_df = pd.read_csv(args.ids)
    ids = ids_df["image_id"].tolist()

    query_df = pd.read_csv(args.query_csv)
    gallery_df = pd.read_csv(args.gallery_csv)

    n_query = len(query_df)
    query_ids = ids[:n_query]
    gallery_ids = ids[n_query:]
    assert query_ids == query_df["image_id"].tolist(), "Порядок query не совпадает!"
    assert gallery_ids == gallery_df["image_id"].tolist(), "Порядок gallery не совпадает!"

    query_emb = embeddings[:n_query]
    gallery_emb = embeddings[n_query:]

    index = VectorIndex(backend=cfg.search.backend)
    index.build(gallery_emb, gallery_ids)

    top_k = cfg.search.top_k
    scores, idxs = index.search(query_emb, top_k=top_k)
    matched_ids = index.get_ids(idxs)

    # submission.csv: query_id, gallery_id_1..10
    sub_rows = []
    for qid, cand_ids in zip(query_ids, matched_ids):
        row = [qid] + cand_ids + [""] * (top_k - len(cand_ids))
        sub_rows.append(row)
    sub_cols = ["query_id"] + [f"gallery_id_{i+1}" for i in range(top_k)]
    pd.DataFrame(sub_rows, columns=sub_cols).to_csv(args.submission_out, index=False)

    # candidates.csv: query_id, gallery_id, confidence (с порогом отказа)
    cand_rows = []
    threshold = cfg.search.reject_threshold
    for qid, cand_ids, cand_scores in zip(query_ids, matched_ids, scores):
        best_score = float(cand_scores[0]) if len(cand_scores) else -1.0
        if best_score < threshold:
            continue  # режим отказа: пустой ответ для данного query
        for gid, score in zip(cand_ids, cand_scores):
            if float(score) < threshold:
                continue
            cand_rows.append((qid, gid, float(score)))

    pd.DataFrame(cand_rows, columns=["query_id", "gallery_id", "confidence"]) \
        .to_csv(args.candidates_out, index=False)

    print(f"submission.csv -> {args.submission_out}")
    print(f"candidates.csv -> {args.candidates_out}")


if __name__ == "__main__":
    main()
