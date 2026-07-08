# src/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=512):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                 rotate=(-15, 15), p=0.4),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=512):
    return A.Compose([
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
        ToTensorV2(),
    ])