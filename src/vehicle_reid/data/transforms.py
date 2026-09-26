import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _color_jitter(p: float = 0.8):
    # Для ReID цвет машины — ключевой признак: hue/saturation слабые, без grayscale.
    return A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02, p=p)


def _geometry(size: int, scale: tuple[float, float]):
    return [
        A.RandomResizedCrop(size=(size, size), scale=scale),  # albumentations >= 1.4
        A.HorizontalFlip(p=0.5),
    ]


def _finalize():
    return [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]


# --------------------------------------------------------------- стадия 1 (DINO)


def build_global_crop_transforms(size: int = 224) -> tuple[A.Compose, A.Compose]:
    """Два асимметричных глобальных трансформа, как в DINO."""
    t1 = A.Compose(
        _geometry(size, (0.4, 1.0)) + [_color_jitter(), A.GaussianBlur(sigma_limit=(0.1, 2.0), p=1.0)] + _finalize()
    )
    t2 = A.Compose(
        _geometry(size, (0.4, 1.0))
        + [_color_jitter(), A.GaussianBlur(sigma_limit=(0.1, 2.0), p=0.1), A.Solarize(p=0.2)]
        + _finalize()
    )
    return t1, t2


def build_local_crop_transforms(size: int = 96) -> A.Compose:
    return A.Compose(
        _geometry(size, (0.05, 0.4)) + [_color_jitter(), A.GaussianBlur(sigma_limit=(0.1, 2.0), p=0.5)] + _finalize()
    )


# -------------------------------------------------- стадия 2 (contrastive/ReID)


def build_train_transforms(image_size, erasing_prob: float = 0.5) -> A.Compose:
    """Мягкие ReID-аугментации: без агрессивного кропа, чтобы не терять геометрию машины.

    Random Erasing (как в BoT/TransReID) — главный регуляризатор против окклюзий.
    """
    h, w = _hw(image_size)
    return A.Compose([
        A.Resize(height=int(h * 1.12), width=int(w * 1.12)),
        A.RandomCrop(height=h, width=w),
        A.HorizontalFlip(p=0.5),
        _color_jitter(p=0.5),
        A.GaussNoise(p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.15, 0.35),
            hole_width_range=(0.15, 0.35),
            fill="random_uniform",
            p=erasing_prob,
        ),
        ToTensorV2(),
    ])


def build_eval_transforms(image_size) -> A.Compose:
    h, w = _hw(image_size)
    return A.Compose([A.Resize(height=h, width=w)] + _finalize())


def _hw(image_size) -> tuple[int, int]:
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) == 1:
        return int(image_size[0]), int(image_size[0])
    return int(image_size[0]), int(image_size[1])
