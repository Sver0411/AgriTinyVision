from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from agritinyvision.image_contract import MEAN, STD


def preprocessing(size):
    # Direct square resize is deliberate, and part of the Gateway contract.
    return T.Compose(
        [
            T.Resize((size, size), interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )


def training_transform(size, a):
    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=(a["crop_min"], 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(a["horizontal_flip"]),
            T.RandomRotation(a["rotation"]),
            T.ColorJitter(
                brightness=a["brightness"], contrast=a["contrast"], saturation=a["saturation"], hue=0
            ),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
