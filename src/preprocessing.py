# src/preprocessing.py
import cv2
import json
import numpy as np
import torch
from pathlib import Path


def load_intensity_stats(stats_path="data/reference/intensity_stats.json"):
    with open(stats_path) as f:
        return json.load(f)


def correct_vignette(img, reference_path=None):
    if reference_path and Path(reference_path).exists():
        ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        ref = np.clip(ref, 1, 255)
        img = cv2.divide(img.astype(np.float32), ref, scale=255)
    else:
        background = cv2.GaussianBlur(img, (101, 101), 0).astype(np.float32)
        background = np.clip(background, 1, 255)
        img = cv2.divide(img.astype(np.float32), background, scale=255)
    return np.clip(img, 0, 255).astype(np.uint8)


def normalize_to_training_distribution(img, target_mean, target_std):
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    img = img * target_std + target_mean
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_clahe(img, clip_limit=3.0, tile_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img)

def preprocess_for_instance_seg(img_path, stats_path=None, reference_path=None, img_size=512):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img = correct_vignette(img, reference_path)

    if stats_path:
        stats = load_intensity_stats(stats_path)
        img = normalize_to_training_distribution(img, stats['mean'], stats['std'])

    img = apply_clahe(img)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img = cv2.resize(img, (img_size, img_size))

    img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    tensor = torch.FloatTensor(img_3ch).permute(2, 0, 1) / 255.0
    return tensor
