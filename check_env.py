# scripts/check_env.py
import numpy as np
import torch
import cv2
import albumentations as A
import faiss
import sklearn
import timm

print("numpy:", np.__version__)
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
print("opencv:", cv2.__version__)
print("albumentations:", A.__version__)
print("faiss:", faiss.__version__)
print("sklearn:", sklearn.__version__)
print("timm:", timm.__version__)

# Быстрый smoke-test на реальных операциях, чтобы поймать ABI-конфликты
x = np.random.rand(3, 224, 224).astype(np.float32)
t = torch.from_numpy(x)
print("torch<->numpy OK:", t.shape)

img = (x.transpose(1, 2, 0) * 255).astype(np.uint8)
resized = cv2.resize(img, (128, 128))
print("opencv<->numpy OK:", resized.shape)

index = faiss.IndexFlatIP(512)
index.add(np.random.rand(10, 512).astype(np.float32))
print("faiss<->numpy OK:", index.ntotal)
