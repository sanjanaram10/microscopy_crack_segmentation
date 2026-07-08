import json
import random
import shutil
import os
from pathlib import Path
from collections import defaultdict

def split_dataset(
    coco_json_path,
    images_src_dir,
    output_dir,
    splits=(0.7, 0.2, 0.1),
    seed=42
):
    """
    Splits CVAT COCO export into train/val/test.
    Copies images into the correct subdirectory.
    """
    assert sum(splits) == 1.0, "Splits must sum to 1.0"
    
    with open(coco_json_path) as f:
        coco = json.load(f)
    
    images = coco['images'].copy()
    random.seed(seed)
    random.shuffle(images)
    
    n = len(images)
    n_train = int(n * splits[0])
    n_val   = int(n * splits[1])
    
    split_images = {
        'train': images[:n_train],
        'val':   images[n_train:n_train + n_val],
        'test':  images[n_train + n_val:]
    }
    
    # Build annotation lookup
    ann_by_image = defaultdict(list)
    for ann in coco['annotations']:
        ann_by_image[ann['image_id']].append(ann)
    
    for split_name, split_imgs in split_images.items():
        # Create directories
        img_out_dir = Path(output_dir) / split_name / 'images'
        img_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Gather annotations for this split
        image_ids = {img['id'] for img in split_imgs}
        annotations = [
            a for a in coco['annotations']
            if a['image_id'] in image_ids
        ]
        
        # Write COCO JSON
        split_coco = {
            'images':      split_imgs,
            'annotations': annotations,
            'categories':  coco['categories']
        }
        json_path = Path(output_dir) / split_name / f'{split_name}.json'
        with open(json_path, 'w') as f:
            json.dump(split_coco, f, indent=2)
        
        # Copy images
        for img_info in split_imgs:
            src = Path(images_src_dir) / img_info['file_name']
            dst = img_out_dir / img_info['file_name']
            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"WARNING: image not found: {src}")
        
        print(f"{split_name:6s}: {len(split_imgs):4d} images, "
              f"{len(annotations):5d} annotations")


if __name__ == "__main__":
    split_dataset(
        coco_json_path="dataset/raw/annotations/instances_default.json",
        images_src_dir="dataset/raw/images/default",
        output_dir="dataset/processed",
        splits=(0.7, 0.2, 0.1),
        seed=42
    )