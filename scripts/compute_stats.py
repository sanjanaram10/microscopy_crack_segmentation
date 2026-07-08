import cv2
import json
import numpy as np
from pathlib import Path

def compute_intensity_stats(images_dir, output_path):
    """
    Computes per-channel mean and std across all training images.
    Used at inference time to normalize new images to training distribution.
    """
    image_paths = list(Path(images_dir).glob("*.png")) + \
                  list(Path(images_dir).glob("*.jpg"))
    
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    
    means, stds = [], []
    histograms  = []
    
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        means.append(float(img.mean()))
        stds.append(float(img.std()))
        
        hist, _ = np.histogram(img.ravel(), bins=256,
                               range=(0, 255), density=True)
        histograms.append(hist.tolist())
    
    mean_histogram = np.mean(histograms, axis=0).tolist()
    
    stats = {
        "num_images":      len(image_paths),
        "mean":            float(np.mean(means)),
        "std":             float(np.mean(stds)),
        "mean_per_image":  means,
        "std_per_image":   stds,
        "mean_histogram":  mean_histogram,  # used for histogram matching
        "histogram_bins":  list(range(256))
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Computed stats over {len(image_paths)} images")
    print(f"  Mean intensity : {stats['mean']:.2f}")
    print(f"  Std  intensity : {stats['std']:.2f}")
    print(f"  Saved to       : {output_path}")
    return stats


if __name__ == "__main__":
    compute_intensity_stats(
        images_dir="data/processed/train/images",
        output_path="data/reference/intensity_stats.json"
    )