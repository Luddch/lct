from __future__ import annotations
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

import os

from src.vehicle_reid.config import load_serving_config
from src.vehicle_reid.api.service import ReIDService
from src.vehicle_reid.api.schemas import SearchResponse, SearchResultItem
from src.vehicle_reid.utils.bbox import BBoxValidationError

app = FastAPI(title="Vehicle ReID Service")

CFG = load_serving_config(os.environ.get("REID_CONFIG", "configs/serving.yaml"))
SERVICE = ReIDService(CFG, weights_path=os.environ.get("REID_WEIGHTS", CFG.weights))


@app.on_event("startup")
def load_gallery():
    """Опционально предзагружаем эмбеддинги галереи, если они посчитаны заранее."""
    try:
        gallery_emb = np.load("outputs/gallery_embeddings.npy")
        gallery_ids = pd.read_csv("outputs/gallery_ids.csv")["image_id"].tolist()
        SERVICE.load_gallery_index(gallery_emb, gallery_ids)
        print("Gallery index loaded.")
    except FileNotFoundError:
        print("Gallery embeddings not found yet — используйте /gallery/build.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    w: float = Form(...),
    h: float = Form(...),
):
    """Этап получения + обработки: возвращает эмбеддинг."""
    image_bytes = await file.read()
    image = SERVICE.read_image(image_bytes)
    try:
        crop = SERVICE.validate_and_crop(image, x, y, w, h)
    except BBoxValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    embedding = SERVICE.extract_embedding(crop)
    return {"embedding_dim": len(embedding), "embedding": embedding.tolist()}


@app.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    w: float = Form(...),
    h: float = Form(...),
    top_k: int = Form(10),
):
    """Полный цикл: получение -> обработка -> анализ -> результат."""
    image_bytes = await file.read()
    image = SERVICE.read_image(image_bytes)
    try:
        crop = SERVICE.validate_and_crop(image, x, y, w, h)
    except BBoxValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    embedding = SERVICE.extract_embedding(crop)

    if SERVICE.index is None:
        raise HTTPException(status_code=503, detail="Индекс галереи не загружен")

    candidates = SERVICE.search(embedding, top_k=top_k)
    filtered, rejected = SERVICE.apply_reject_threshold(candidates)

    return SearchResponse(
        results=[SearchResultItem(gallery_id=gid, confidence=score) for gid, score in filtered],
        rejected=rejected,
    )
