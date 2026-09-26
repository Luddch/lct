from dataclasses import dataclass, field
from typing import List


@dataclass
class ReIDTrainingConfig:
    """Стадия 2: contrastive + ID-loss на размеченном датасете.

    Значения по умолчанию подобраны под ~16-24 ГБ VRAM и ViT-L/16 с заморозкой
    нижних блоков. Для ViT-B можно поднять micro_batch_size и снять freeze_blocks.
    """

    # --- Модель ---
    backbone: str = "vit_large_patch16_dinov3.lvd1689m"
    pretrained: bool = True
    dino_weights: str | None = None       # путь к teacher_backbone.pt со стадии 1
    embedding_dim: int = 0                # 0 = размерность бэкбона без редукции
    pooling: str = "cls"                  # cls | gem
    neck: str = "bnneck"
    drop_path_rate: float = 0.1
    grad_checkpointing: bool = True
    freeze_blocks: int = 12               # сколько нижних блоков заморозить (ViT-L: 24 всего)
    freeze_backbone_steps: int = 100      # прогрев головы перед разморозкой бэкбона

    # --- Данные ---
    image_size: List[int] = field(default_factory=lambda: [224, 224])
    bbox_padding: float = 0.10
    val_ratio: float = 0.15
    n_query_per_id: int = 1
    erasing_prob: float = 0.5
    num_workers: int = 8

    # --- Батчи (в изображениях; каждое даёт n_views аугментаций) ---
    micro_batch_size: int = 8
    accumulation_steps: int = 4
    n_views: int = 2
    p_per_batch: int = 0                  # >0 включает PK-сэмплер: P ID x K картинок
    k_per_id: int = 4

    # --- Оптимизация ---
    epochs: int = 20
    lr_backbone: float = 1e-5
    lr_head: float = 5e-4
    lr_min_ratio: float = 0.01
    warmup_steps: int = 100
    weight_decay: float = 0.05
    gradient_clip: float = 5.0
    amp_dtype: str = "bfloat16"

    # --- Лоссы ---
    supcon_weight: float = 1.0
    supcon_temperature: float = 0.07
    queue_size: int = 8192
    use_momentum_encoder: bool = True     # MoCo-style ключи; стабилизирует очередь
    momentum_start: float = 0.99
    momentum_end: float = 0.999
    id_weight: float = 1.0
    use_arcface: bool = True
    arcface_scale: float = 30.0
    arcface_margin: float = 0.30
    label_smoothing: float = 0.1
    center_weight: float = 0.0

    # --- Валидация / чекпоинты ---
    eval_every: int = 1                   # в эпохах
    eval_batch_size: int = 64
    early_stop_patience: int = 6
    log_interval: int = 10
    keep_last_checkpoints: int = 2
    checkpoint_dir: str = "./checkpoints/reid"
    seed: int = 42

    def __post_init__(self):
        if self.micro_batch_size < 1 or self.accumulation_steps < 1:
            raise ValueError("micro_batch_size и accumulation_steps должны быть >= 1")
        if self.n_views < 1:
            raise ValueError("n_views должен быть >= 1")
        if self.p_per_batch and self.p_per_batch * self.k_per_id != self.micro_batch_size:
            raise ValueError(
                f"PK-сэмплер: p_per_batch*k_per_id ({self.p_per_batch * self.k_per_id}) "
                f"должен равняться micro_batch_size ({self.micro_batch_size})"
            )
        if self.amp_dtype not in ("bfloat16", "float16", "none"):
            raise ValueError("amp_dtype: bfloat16 | float16 | none")

    @property
    def effective_batch(self) -> int:
        return self.micro_batch_size * self.accumulation_steps * self.n_views
