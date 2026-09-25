import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _color_jitter():
    # Для ReID цвет машины — ключевой признак: hue/saturation слабые, без grayscale.
    return A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02, p=0.8)


def _geometry(size: int, scale: tuple[float, float]):
    return [
        A.RandomResizedCrop(size=(size, size), scale=scale),  # albumentations >= 1.4
        A.HorizontalFlip(p=0.5),
    ]


def _finalize():
    return [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]


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