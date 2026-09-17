from __future__ import annotations
import numpy as np
import torch


class VectorIndex:
    """
    Точный косинусный поиск.
    backend="torch": matmul на GPU (быстро для галерей до ~10^6 при float16).
    backend="faiss_cpu": faiss.IndexFlatIP на CPU (fallback, без GPU-зависимости).
    """

    def __init__(self, backend: str = "torch", device: str = "cuda"):
        self.backend = backend
        self.device = device if torch.cuda.is_available() else "cpu"
        self.embeddings = None
        self.ids: list[str] = []
        self._faiss_index = None

    def build(self, embeddings: np.ndarray, ids: list[str]):
        embeddings = embeddings.astype(np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        self.ids = list(ids)

        if self.backend == "faiss_cpu":
            import faiss
            self._faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            self._faiss_index.add(embeddings)
            self.embeddings = embeddings
        else:
            self.embeddings = torch.from_numpy(embeddings).to(self.device)

    def search(self, query_embeddings: np.ndarray, top_k: int = 10):
        query_embeddings = query_embeddings.astype(np.float32)
        query_embeddings = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12)

        if self.backend == "faiss_cpu":
            scores, idxs = self._faiss_index.search(query_embeddings, top_k)
            return scores, idxs
        else:
            q = torch.from_numpy(query_embeddings).to(self.device)
            sims = q @ self.embeddings.T  # cosine similarity (векторы нормированы)
            scores, idxs = torch.topk(sims, k=min(top_k, sims.shape[1]), dim=1)
            return scores.cpu().numpy(), idxs.cpu().numpy()

    def get_ids(self, idxs: np.ndarray):
        return [[self.ids[i] for i in row] for row in idxs]
