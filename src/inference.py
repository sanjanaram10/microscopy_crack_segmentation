import torch
from pathlib import Path
from src.config import Config
from src.model import build_model
from src.preprocessing import preprocess_for_instance_seg
from src.postprocessing import get_per_crack_stats


def load_model(checkpoint_path, cfg):
    model = build_model(cfg.NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    
    # Handle both raw state_dict and full checkpoint saves
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(cfg.DEVICE)
    model.eval()
    return model


def predict_image(model, img_path, cfg, normalize_domain=False):
    """
    normalize_domain: set True when running on new lighter images
                      to shift intensity to training distribution
    """
    stats_path = "data/reference/intensity_stats.json" \
                 if normalize_domain else None
    
    img = preprocess_for_instance_seg(
        img_path,
        stats_path=stats_path,
        img_size=cfg.IMG_SIZE
    )
    
    df, masks = get_per_crack_stats(
        model, img,
        pixel_size_mm=cfg.PIXEL_SIZE_MM,
        confidence_threshold=cfg.CONFIDENCE_THRESHOLD,
        min_area_px=cfg.MIN_INSTANCE_AREA_PX,
        device=cfg.DEVICE
    )
    
    return df, masks