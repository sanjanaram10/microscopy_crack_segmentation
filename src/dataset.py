import torch
from torch.utils.data import Dataset
import json
import cv2
from src.preprocessing import preprocess_for_instance_seg
from pycocotools import mask as coco_mask
from src.config import Config as cfg
import numpy as np

class CrackDataset(Dataset):
    def __init__(self, img_dir, coco_json_path, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(coco_json_path) as f:
            coco = json.load(f)

        self.images = coco['images']
        self.ann_by_image = {}
        for ann in coco['annotations']:
            iid = ann['image_id']
            self.ann_by_image.setdefault(iid, []).append(ann)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = f"{self.img_dir}/{img_info['file_name']}"

        # Returns tensor [3, H, W] float32 in [0,1]
        img_tensor = preprocess_for_instance_seg(img_path)

        annotations = self.ann_by_image.get(img_info['id'], [])

        masks, boxes, labels = [], [], []
        for ann in annotations:
            rle = coco_mask.frPyObjects(
                ann['segmentation'],
                img_info['height'],
                img_info['width']
            )
            m = coco_mask.decode(rle).squeeze()
            m = cv2.resize(
                m.astype("uint8"),
                (cfg.IMG_SIZE, cfg.IMG_SIZE),
                interpolation=cv2.INTER_NEAREST
            )

            scale_x = cfg.IMG_SIZE / img_info['width']
            scale_y = cfg.IMG_SIZE / img_info['height']

            x, y, w, h = ann['bbox']
            x1 = max(0, x) * scale_x
            y1 = max(0, y) * scale_y
            x2 = min(img_info['width'], x + w) * scale_x
            y2 = min(img_info['height'], y + h) * scale_y

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(1)
                masks.append(torch.as_tensor(m, dtype=torch.uint8))

        # Apply image-only transforms
        if self.transforms is not None:
            # Convert tensor [3, H, W] float [0,1] → numpy [H, W, 3] uint8
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            transformed = self.transforms(image=img_np)
            img_tensor = transformed['image']  # ToTensorV2 converts back to [3, H, W] tensor

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "masks": torch.stack(masks)
                    if masks
                    else torch.zeros((0, cfg.IMG_SIZE, cfg.IMG_SIZE), dtype=torch.uint8),
            "image_id": torch.tensor([img_info['id']])
        }

        return img_tensor, target

    def __len__(self):
        return len(self.images)