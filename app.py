import io
import os
import cv2
import torch
import tempfile
import numpy as np
import base64
import asyncio
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse

from src.model import build_model
from src.preprocessing import preprocess_for_instance_seg
from src.config import Config

app = FastAPI()
cfg = Config()

# Load model once at startup
model = build_model(cfg.NUM_CLASSES)
checkpoint = torch.load("checkpoints/model_best.pth", map_location=cfg.DEVICE)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.to(cfg.DEVICE)
model.eval()


# ── Post-processing helpers ──────────────────────────────────────────────────

def mask_nms(masks, scores, iou_threshold=0.1):
    if len(masks) == 0:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        for j in order:
            if j <= i or j in suppressed:
                continue
            intersection = (masks[i] & masks[j]).sum()
            union = (masks[i] | masks[j]).sum()
            iou = intersection / union if union > 0 else 0
            if iou > iou_threshold:
                suppressed.add(j)
    return kept


def merge_overlapping_masks(masks, scores, overlap_threshold=0.2):
    if len(masks) == 0:
        return [], []

    used = set()
    merged_masks = []
    merged_scores = []

    for i in range(len(masks)):
        if i in used:
            continue
        current_mask = masks[i].copy()
        current_score = scores[i]
        changed = True

        # Keep merging until no more overlaps found
        while changed:
            changed = False
            for j in range(len(masks)):
                if j in used or j == i:
                    continue

                intersection = (current_mask & masks[j]).sum()
                union = (current_mask | masks[j]).sum()
                smaller_area = min(current_mask.sum(), masks[j].sum())

                # Check both overlap ratio and IoU
                overlap_ratio = intersection / smaller_area if smaller_area > 0 else 0
                iou = intersection / union if union > 0 else 0

                if overlap_ratio > overlap_threshold or iou > 0.1:
                    current_mask = (current_mask | masks[j]).astype(np.uint8)
                    current_score = max(current_score, scores[j])
                    used.add(j)
                    changed = True  # loop again to catch chain merges

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        used.add(i)

    return merged_masks, merged_scores


def calculate_crack_metrics(binary_mask):
    area = int(binary_mask.sum())
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    else:
        x, y, w, h = 0, 0, 0, 0
    if area > 0:
        M = cv2.moments(binary_mask)
        cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else x + w // 2
        cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else y + h // 2
    else:
        cx, cy = x + w // 2, y + h // 2
    return {
        "area_px":      area,
        "perimeter_px": round(perimeter, 1),
        "bbox":         [x, y, w, h],
        "centroid":     [cx, cy]
    }


# ── Multi-scale inference helpers ────────────────────────────────────────────

def run_inference_on_region(img_bgr, model, device, cfg):
    """Preprocess a region and run inference, return masks at region size."""
    orig_h, orig_w = img_bgr.shape[:2]

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        temp_path = tmp.name

    try:
        img_tensor = preprocess_for_instance_seg(temp_path)
        with torch.no_grad():
            predictions = model([img_tensor.to(device)])
    finally:
        os.unlink(temp_path)

    pred = predictions[0]
    masks, scores = [], []

    for score, mask in zip(pred['scores'], pred['masks']):
        if score < cfg.CONFIDENCE_THRESHOLD:
            continue
        binary = (mask.squeeze() > 0.5).cpu().numpy().astype(np.uint8)
        binary = cv2.resize(binary, (orig_w, orig_h),
                           interpolation=cv2.INTER_NEAREST)
        masks.append(binary)
        scores.append(float(score))

    return masks, scores


def inference_with_crops(img_bgr, model, device, cfg):
    """Run inference on full image and quadrant crops, merge results."""
    h, orig_w = img_bgr.shape[:2]
    all_masks, all_scores = [], []

    # Full image
    masks, scores = run_inference_on_region(img_bgr, model, device, cfg)
    all_masks.extend(masks)
    all_scores.extend(scores)

    # Four quadrant crops
    mid_h, mid_w = h // 2, orig_w // 2
    crops = [
        (img_bgr[:mid_h, :mid_w],  0,     0),      # top-left
        (img_bgr[:mid_h, mid_w:],  0,     mid_w),  # top-right
        (img_bgr[mid_h:, :mid_w],  mid_h, 0),      # bottom-left
        (img_bgr[mid_h:, mid_w:],  mid_h, mid_w),  # bottom-right
    ]

    for crop, y_offset, x_offset in crops:
        crop_masks, crop_scores = run_inference_on_region(crop, model, device, cfg)
        crop_h, crop_w = crop.shape[:2]

        for mask in crop_masks:
            # Place crop mask back into full image coordinate space
            full_mask = np.zeros((h, orig_w), dtype=np.uint8)
            resized = cv2.resize(mask, (crop_w, crop_h),
                                interpolation=cv2.INTER_NEAREST)
            full_mask[y_offset:y_offset + crop_h,
                      x_offset:x_offset + crop_w] = resized
            all_masks.append(full_mask)

        all_scores.extend(crop_scores)

    return all_masks, all_scores


# ── Per-file inference ───────────────────────────────────────────────────────

async def process_single(file):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig_h, orig_w = img_bgr.shape[:2]

    # Multi-scale inference — full image + four quadrant crops
    raw_masks, raw_scores = inference_with_crops(img_bgr, model, cfg.DEVICE, cfg)

    # Suppress duplicates then merge remaining overlaps
    kept_indices = mask_nms(raw_masks, raw_scores, iou_threshold=0.1)
    nms_masks  = [raw_masks[i]  for i in kept_indices]
    nms_scores = [raw_scores[i] for i in kept_indices]

    final_masks, final_scores = merge_overlapping_masks(
        nms_masks, nms_scores, overlap_threshold=0.2
    )

    colors = [
        (255, 0, 0),   (0, 255, 0),   (0, 0, 255),
        (255, 165, 0), (128, 0, 128), (0, 255, 255),
        (255, 20, 147),(0, 128, 0),   (255, 215, 0),
    ]

    result_img = img_bgr.copy()
    cracks = []

    for crack_idx, (binary, score) in enumerate(zip(final_masks, final_scores)):
        metrics = calculate_crack_metrics(binary)
        metrics["score"] = round(score, 3)
        metrics["id"] = crack_idx
        color = colors[crack_idx % len(colors)]
        result_img[binary == 1] = color
        cracks.append(metrics)

    _, buffer = cv2.imencode('.png', result_img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    _, orig_buffer = cv2.imencode('.png', img_bgr)
    orig_b64 = base64.b64encode(orig_buffer).decode('utf-8')

    return {
        "filename":     file.filename,
        "original_b64": orig_b64,
        "image_b64":    img_b64,
        "crack_count":  len(cracks),
        "cracks":       cracks
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc.errors()), status_code=422)


@app.post("/analyze")
async def analyze_image(files: List[UploadFile] = File(...)):
    results = await asyncio.gather(*[process_single(f) for f in files])
    return JSONResponse({"results": list(results)})


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")