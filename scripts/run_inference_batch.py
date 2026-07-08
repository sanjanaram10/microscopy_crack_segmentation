import torch
import pandas as pd
from pathlib import Path
from src.config import Config
from src.model import build_model
from src.inference import load_model, predict_image
import cv2
import numpy as np

def save_overlay(img_path, masks, scores, output_path):
    """Save predicted masks overlaid on original image"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    colors = (np.random.rand(len(masks), 3) * 255).astype(np.uint8)
    
    for mask, score, color in zip(masks, scores, colors):
        binary = (mask > 0.5).astype(np.uint8)
        overlay[binary == 1] = (
            overlay[binary == 1] * 0.4 + color * 0.6
        ).astype(np.uint8)
    
    cv2.imwrite(str(output_path), overlay)


def run_batch(input_dir, checkpoint_path, output_dir):
    cfg = Config()
    model = load_model(checkpoint_path, cfg)
    
    input_paths = list(Path(input_dir).glob("*.png")) + \
                  list(Path(input_dir).glob("*.jpg"))
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    overlay_dir = Path(output_dir) / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    
    all_stats = []
    
    for img_path in input_paths:
        print(f"Processing: {img_path.name}")
        
        df, masks = predict_image(model, img_path, cfg)
        df['image'] = img_path.name
        all_stats.append(df)
        
        # Save per-image CSV
        df.to_csv(
            Path(output_dir) / f"{img_path.stem}_cracks.csv",
            index=False
        )
        
        # Save overlay visualization
        if len(masks) > 0:
            scores = df['confidence'].values
            save_overlay(
                img_path, masks, scores,
                overlay_dir / f"{img_path.stem}_overlay.png"
            )
        
        print(f"  Found {len(df)} cracks")
    
    # Consolidated CSV across all images
    combined = pd.concat(all_stats, ignore_index=True)
    combined.to_csv(Path(output_dir) / "all_crack_statistics.csv", index=False)
    print(f"\nDone. Results saved to {output_dir}")
    print(combined.describe())


if __name__ == "__main__":
    run_batch(
        input_dir="new_images/",
        checkpoint_path="checkpoints/model_best.pth",
        output_dir="outputs/crack_statistics"
    )