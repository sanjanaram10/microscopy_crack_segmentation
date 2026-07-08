import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

def visualize_split(split_name, data_dir, n_samples=5, output_dir="outputs/qa"):
    json_path  = Path(data_dir) / split_name / f"{split_name}.json"
    images_dir = Path(data_dir) / split_name / "images"
    out_dir    = Path(output_dir) / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    coco = COCO(str(json_path))
    img_ids = coco.getImgIds()
    sample  = random.sample(img_ids, min(n_samples, len(img_ids)))
    
    for img_id in sample:
        img_info = coco.loadImgs(img_id)[0]
        img_path = images_dir / img_info['file_name']
        img      = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img_rgb  = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns    = coco.loadAnns(ann_ids)
        
        # Draw each instance in a different color
        overlay = img_rgb.copy()
        colors  = plt.cm.tab20(np.linspace(0, 1, max(len(anns), 1)))
        
        for ann, color in zip(anns, colors):
            color_uint8 = (np.array(color[:3]) * 255).astype(np.uint8)
            
            # Decode mask
            rle  = coco_mask.frPyObjects(
                ann['segmentation'],
                img_info['height'],
                img_info['width']
            )
            m = coco_mask.decode(rle).squeeze()
            
            # Colored mask overlay
            overlay[m == 1] = (
                overlay[m == 1] * 0.5 + color_uint8 * 0.5
            ).astype(np.uint8)
            
            # Bounding box
            x, y, w, h = [int(v) for v in ann['bbox']]
            cv2.rectangle(overlay, (x,y), (x+w, y+h),
                         color_uint8.tolist(), 2)
            
            # Instance ID
            cv2.putText(overlay, str(ann['id']),
                       (x+2, y+15), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color_uint8.tolist(), 1)
        
        # Plot with stats
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        axes[1].imshow(overlay)
        axes[1].set_title(
            f"{img_info['file_name']} — {len(anns)} instances\n"
            f"Areas: {[a['area'] for a in anns]}"
        )
        axes[1].axis('off')
        
        plt.tight_layout()
        out_path = out_dir / f"qa_{img_id}_{img_info['file_name']}"
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")
    
    # Print summary statistics
    all_ann_ids = coco.getAnnIds()
    all_anns    = coco.loadAnns(all_ann_ids)
    areas       = [a['area'] for a in all_anns]
    
    print(f"\n{'='*40}")
    print(f"Split: {split_name}")
    print(f"  Images      : {len(img_ids)}")
    print(f"  Instances   : {len(all_anns)}")
    print(f"  Avg per img : {len(all_anns)/len(img_ids):.1f}")
    print(f"  Area min    : {min(areas):.0f} px")
    print(f"  Area mean   : {np.mean(areas):.0f} px")
    print(f"  Area max    : {max(areas):.0f} px")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        visualize_split(split, data_dir="data/processed", n_samples=5)