import torch
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from src.config import Config
from src.model import build_model
from src.dataset import CrackDataset
from src.transforms import get_val_transforms
import numpy as np

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
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=cfg.DEVICE)
    )
    model.to(cfg.DEVICE)
    model.eval()
    
    coco_gt   = COCO(f"{cfg.DATA_DIR}/{split}/{split}.json")
    coco_preds = []
    
    with torch.no_grad():
        for imgs, targets in loader:
            imgs  = [img.to(cfg.DEVICE) for img in imgs]
            preds = model(imgs)
            
            for pred, target in zip(preds, targets):
                img_id = int(target['image_id'].item())
                
                for score, label, mask in zip(
                    pred['scores'], pred['labels'], pred['masks']
                ):
                    if score < cfg.CONFIDENCE_THRESHOLD:
                        continue
                    
                    binary = (mask.squeeze() > 0.5).cpu().numpy().astype(np.uint8)
                    
                    # RLE encode mask for COCO eval
                    from pycocotools import mask as coco_mask
                    rle = coco_mask.encode(np.asfortranarray(binary))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    
                    coco_preds.append({
                        'image_id':    img_id,
                        'category_id': int(label.item()),
                        'segmentation': rle,
                        'score':       float(score.item())
                    })
    
    if not coco_preds:
        print("No predictions above threshold — check confidence_threshold")
        return
    
    coco_dt  = coco_gt.loadRes(coco_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # summarize() prints:
    # AP @[IoU=0.50:0.95], AP @[IoU=0.50], AP @[IoU=0.75]
    # AP small/medium/large, AR @[maxDets=1/10/100]


if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/model_best.pth"
    evaluate(checkpoint, split='test')