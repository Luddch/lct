from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # --- Модель ---
    model_name: str = "vit_large_patch16_dinov3.lvd1689m"
    drop_path_rate: float = 0.1
    grad_checkpointing: bool = True
    out_dim: int = 16384            # число прототипов DINO-головы (для небольшого датасета 65536 избыточно)
    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256

    # --- Данные ---
    global_size: int = 224
    local_size: int = 96
    n_local_crops: int = 6
    num_workers: int = 8

    # --- Оптимизация ---
    epochs: int = 30
    lr_backbone: float = 2e-5
    lr_head: float = 2e-4
    lr_min_ratio: float = 0.01      # lr_min = base_lr * lr_min_ratio
    warmup_steps: int = 100         # в шагах оптимизатора
    weight_decay: float = 0.04
    gradient_clip: float = 3.0
    backbone_freeze_steps: int = 200    # сначала учим только голову
    freeze_last_layer_steps: int = 100  # как в DINO: last layer заморожен в начале

    # --- Batch sizes (в МАШИНАХ, не в картинках) ---
    gpu_batch_size: int = 1
    batch_size_start: int = 4
    batch_size_end: int = 16
    batch_size_warmup_steps: int = 300

    # --- DINO ---
    student_temp: float = 0.1
    teacher_temp: float = 0.06
    warmup_teacher_temp: float = 0.04
    teacher_temp_warmup_steps: int = 300
    center_momentum: float = 0.9
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0

    # --- Логирование / чекпоинты ---
    log_interval: int = 10
    save_interval: int = 500
    keep_last_checkpoints: int = 3
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42

    def __post_init__(self):
        assert self.batch_size_start % self.gpu_batch_size == 0
        assert self.batch_size_end % self.gpu_batch_size == 0
        assert self.batch_size_start <= self.batch_size_end
        assert self.global_size % 16 == 0 and self.local_size % 16 == 0