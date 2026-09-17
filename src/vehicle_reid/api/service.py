from __future__ import annotations
import io
import numpy as np
import cv2
import torch
from PIL import Image

from src.vehicle_reid.utils.bbox import BBox, crop_with_padding, BBoxValidationError
from src.vehicle_reid.data.transforms import build_eval_transforms
from src.vehicle_reid.models.model import VehicleReIDModel
from src.vehicle_reid.search.index import VectorIndex


class ReIDService:
    def __init__(self, cfg, weights_path):
        self.cfg = cfg
        self.device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")

        self.model = VehicleReIDModel(
            cfg.model.backbone, pretrained=False, embedding_dim=cfg.model.embedding_dim,
            num_classes=1, pooling=cfg.model.pooling,
        )

        checkpoint_state = torch.load(weights_path, map_location="cpu")
        model_state = self.model.state_dict()
        filtered_state = {
            k: v for k, v in checkpoint_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        self.model.load_state_dict(filtered_state, strict=False)
        self.model.to(self.device).eval()

        self.transforms = build_eval_transforms(cfg.data.image_size)
        self.index: VectorIndex | None = None

    # ---------- Этап получения ----------
    @staticmethod
    def read_image(file_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(image)

    def validate_and_crop(self, image: np.ndarray, x, y, w, h) -> np.ndarray:
        bbox = BBox(x=x, y=y, w=w, h=h)
        return crop_with_padding(image, bbox, self.cfg.data.bbox_padding)

    # ---------- Этап обработки ----------
    @torch.no_grad()
    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        tensor = self.transforms(image=crop)["image"].unsqueeze(0).to(self.device)
        emb = self.model.get_embedding(tensor)
        return emb.cpu().numpy().astype(np.float32)[0]

    # ---------- Этап анализа ----------
    def load_gallery_index(self, embeddings: np.ndarray, ids: list[str]):
        self.index = VectorIndex(backend=self.cfg.search.backend)
        self.index.build(embeddings, ids)

    def search(self, query_embedding: np.ndarray, top_k: int | None = None):
        assert self.index is not None, "Индекс галереи не загружен"
        top_k = top_k or self.cfg.search.top_k
        scores, idxs = self.index.search(query_embedding[None, :], top_k=top_k)
        gallery_ids = self.index.get_ids(idxs)[0]
        return list(zip(gallery_ids, scores[0].tolist()))

    # ---------- Этап результата ----------
    def apply_reject_threshold(self, candidates: list[tuple[str, float]]):
        threshold = self.cfg.search.reject_threshold
        if not candidates or candidates[0][1] < threshold:
            return [], True
        filtered = [(gid, score) for gid, score in candidates if score >= threshold]
        return filtered, False
