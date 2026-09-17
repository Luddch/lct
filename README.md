python -m scripts.train --config configs/config.yaml
python -m scripts.extract_embeddings --config configs/config.yaml
python -m scripts.build_submission --config configs/config.yaml
python -m uvicorn vehicle_reid.api.main:app --host 0.0.0.0 --port 8001 --app-dir src
