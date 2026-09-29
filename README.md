# Vehicle ReID — двухстадийный пайплайн в стиле TransReID

1. **Стадия 1 — DINO self-distillation** на нескольких (не)размеченных датасетах: бэкбон учится
   вариативности домена без меток.
2. **Стадия 2 — contrastive fine-tuning** на финальном размеченном датасете:
   SupCon с memory-очередью + ArcFace поверх BNNeck. **Batch-hard triplet не используется** —
   негативы берутся из очереди, поэтому качество почти не зависит от размера батча.
3. **Валидация** — `scripts/compare_models.py`: одинаковый сплит, mAP / Rank-k / mINP,
   дельты к baseline, парный бутстрэп, CMC-кривые и примеры выдачи.

```
configs/dino.yaml      стадия 1: список источников + гиперпараметры DINO
configs/reid.yaml      стадия 2: размеченный датасет, лоссы, VRAM-настройки
configs/serving.yaml   инференс: веса, размер входа, параметры поиска
```

## Установка

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Стадия 1: DINO на нескольких датасетах

Источники описываются списком в `configs/dino.yaml` — можно мешать форматы:

```yaml
sources:
  - name: target                 # целевой датасет с bbox и метками
    type: csv
    csv: /data/target/train.csv  # колонки: image_id, x, y, w, h, vehicle_id
    images_dir: /data/target/images
    group_by: id                 # глобальные виды берутся с РАЗНЫХ фото одной машины
    repeat: 2                    # целевой домен видим чаще вспомогательных

  - name: veri_wild              # внешние неразмеченные кропы
    type: folder
    images_dir: /data/unlabeled/veri_wild
    group_by: none

  - name: tracks                 # картинки разложены по папкам-трекам
    type: folder
    images_dir: /data/unlabeled/tracks
    group_by: parent             # группировка по имени папки
```

```bash
python -m scripts.train_dino --config configs/dino.yaml
python -m scripts.train_dino --config configs/dino.yaml --resume auto   # продолжить
```

Результат: `checkpoints/dino/teacher_backbone.pt` (бэкбон EMA-учителя — он стабильнее студента).

## Как применяется bbox

Один общий путь для всех стадий: `sources.groups_from_csv` читает колонки `x, y, w, h` →
кладёт их в `Record.bbox` → `datasets.load_crop` режет кроп (`crop_with_padding`, расширение
рамки на `bbox_padding` с каждой стороны) → и **только потом** идут аугментации и resize.

Все датасеты наследуют один и тот же `_CropLoaderMixin._load`, поэтому кроп одинаково
применяется в `MultiCropDataset` (DINO), `ReIDTrainDataset` (обучение),
`ReIDEvalDataset` (валидация в обучении, `compare_models.py`, `extract_embeddings.py`)
и в API (`ReIDService.validate_and_crop`).

Чтобы случайно не обучиться на целых кадрах, у источника есть `require_bbox: true` —
тогда отсутствие колонок `x,y,w,h` или `use_bbox: false` приведёт к ошибке, а не к тихому
переходу на полные изображения. В логах при старте всегда печатается фактическая статистика:

```
Источник 'target': 1200 групп, 4800 изображений, с bbox 4800/4800 (repeat=1)
Всего: 1200 групп, 4800 изображений, кропов по bbox 4800/4800
```

Если картинки уже являются кропами (внешние неразмеченные датасеты), ставьте
`use_bbox: false` / `type: folder` — тогда берётся изображение целиком, это ожидаемо.

## Стадия 2: contrastive fine-tuning

```bash
python -m scripts.train_reid --config configs/reid.yaml \
    --dino-weights checkpoints/dino/teacher_backbone.pt
```

Что происходит:

* сплит **по ID** (disjoint identities, `val_ratio`), из валидации собирается query/gallery;
* каждое изображение даёт `n_views` аугментаций → позитивы есть даже при маленьком батче;
* **SupCon с очередью** (`queue_size`): негативы накапливаются между шагами, ключи считает
  momentum-энкодер (MoCo-style), его активации не хранятся;
* **ArcFace поверх BNNeck** как ID-loss (contrastive — до BNNeck, классификация — после, как в BoT/TransReID);
* после каждой эпохи — валидация, `best.pt` по mAP, early stopping, история в `val_history.json`.

### Настройки под ограниченную VRAM

| Рычаг | Параметр | Эффект |
|---|---|---|
| Градиентная аккумуляция | `micro_batch_size` / `accumulation_steps` | эффективный батч не влияет на память |
| Gradient checkpointing | `grad_checkpointing: true` | ~40% памяти активаций ценой ~30% скорости |
| Заморозка нижних блоков | `freeze_blocks: 12` | нет градиентов/состояния Adam для половины ViT-L |
| bf16-autocast | `amp_dtype: bfloat16` | веса в fp32, вычисления в bf16 |
| Очередь вместо большого батча | `queue_size: 8192` | 8192 негатива ≈ 32 МБ, а не гигабайты активаций |

Ориентиры: ViT-L/16 @224, `freeze_blocks: 12`, `micro_batch_size: 8`, `n_views: 2` ≈ 18-20 ГБ.
Если не влезает — сначала `micro_batch_size: 4` + `accumulation_steps: 8`, затем
`freeze_blocks: 16`, затем переход на ViT-B (`vit_base_patch16_dinov3.lvd1689m`, `freeze_blocks: 4`).

## Валидация: насколько стало лучше

```bash
python -m scripts.compare_models --config configs/reid.yaml \
    --model "baseline=pretrained" \
    --model "dino=checkpoints/dino/teacher_backbone.pt" \
    --model "finetuned=checkpoints/reid/best.pt" \
    --out outputs/comparison
```

Все модели оцениваются на **одном и том же** query/gallery-сплите (детерминирован `seed` и
`val_ratio` из конфига, ID не пересекаются с train). На выходе `outputs/comparison/`:

* `report.md` — таблица mAP / Rank-1/5/10 / mINP с абсолютными и относительными дельтами;
* парный бутстрэп по query: 95% ДИ разницы и доля бутстрэп-выборок с улучшением.
  Улучшение уверенное, если нижняя граница CI95 > 0 — это защищает от «выигрыша» на шуме
  при маленькой валидации;
* `cmc.png`, `metrics.png` — CMC-кривые и столбчатое сравнение;
* `retrieval/query_*.png` — top-k выдача каждой модели для одних и тех же query
  (зелёная рамка — верный ID, красная — ошибка);
* `results.json` — то же в машинночитаемом виде.

Три точки отсчёта отвечают на разные вопросы: `baseline` — что даёт бэкбон «из коробки»,
`dino` — вклад стадии 1 отдельно, `finetuned` — итог. Полезно также обучить контроль без
стадии 1: `python -m scripts.train_reid --config configs/reid.yaml --no-dino --checkpoint-dir checkpoints/reid_nodino`.

## Инференс и сервис

```bash
python -m scripts.extract_embeddings --config configs/serving.yaml \
    --query-csv data/test_query.csv --gallery-csv data/test_gallery.csv \
    --images-dir data/images --require-bbox     # --no-bbox, если картинки уже кропы
python -m scripts.build_submission --config configs/serving.yaml \
    --query_csv data/test_query.csv --gallery_csv data/test_gallery.csv

REID_WEIGHTS=checkpoints/reid/best.pt python -m uvicorn src.vehicle_reid.api.main:app --host 0.0.0.0 --port 8001
streamlit run ui/app_streamlit.py
```

## Структура

```
scripts/train_dino.py        стадия 1
scripts/train_reid.py        стадия 2
scripts/compare_models.py    сравнение моделей + отчёт
src/vehicle_reid/data/sources.py    описание источников (folder/csv), сборка групп
src/vehicle_reid/data/datasets.py   MultiCrop (DINO), ReIDTrain/Eval, сплиты
src/vehicle_reid/losses/losses.py   DINOLoss, SupConMemoryLoss, ArcFaceHead, CenterLoss
src/vehicle_reid/models/model.py    DINOHead/MultiCropWrapper, VehicleReIDModel (GeM + BNNeck)
src/vehicle_reid/engine/            тренеры обеих стадий, метрики (mAP/CMC/mINP)
```
