import cv2
import sys
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader

from src.config import Config
from src.model import build_model
from src.dataset import CrackDataset
from src.transforms import get_val_transforms


def evaluate(checkpoint_path, split='test'):
    cfg = Config()

    dataset = CrackDataset(
        f"{cfg.DATA_DIR}/{split}/images",
        f"{cfg.DATA_DIR}/{split}/{split}.json",
        transforms=get_val_transforms(cfg.IMG_SIZE)
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=lambda b: tuple(zip(*b)))

    model = build_model(cfg.NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(cfg.DEVICE)
    model.eval()

    coco_gt = COCO(f"{cfg.DATA_DIR}/{split}/{split}.json")
    coco_preds = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(cfg.DEVICE) for img in imgs]
            preds = model(imgs)

            for pred, target in zip(preds, targets):
                img_id = int(target['image_id'].item())

                # Get original image dimensions for mask resizing
                orig_h = coco_gt.imgs[img_id]['height']
                orig_w = coco_gt.imgs[img_id]['width']

                for score, label, mask in zip(
                    pred['scores'], pred['labels'], pred['masks']
                ):
                    if score < cfg.CONFIDENCE_THRESHOLD:
                        continue

                    binary = (mask.squeeze() > 0.5).cpu().numpy().astype(np.uint8)

                    # Resize from model output size (512x512) to original
                    # image dimensions so masks align with GT annotations
                    binary = cv2.resize(binary, (orig_w, orig_h),
                                       interpolation=cv2.INTER_NEAREST)

                    rle = coco_mask.encode(np.asfortranarray(binary))
                    rle['counts'] = rle['counts'].decode('utf-8')

                    coco_preds.append({
                        'image_id':     img_id,
                        'category_id':  int(label.item()),
                        'segmentation': rle,
                        'score':        float(score.item())
                    })

    print(f"Total predictions submitted: {len(coco_preds)}")
    if not coco_preds:
        print("No predictions above threshold — lower cfg.CONFIDENCE_THRESHOLD")
        return

    print(f"Score range: {min(p['score'] for p in coco_preds):.3f} – "
          f"{max(p['score'] for p in coco_preds):.3f}")

    coco_dt = coco_gt.loadRes(coco_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/model_best.pth"
    evaluate(checkpoint, split='test')