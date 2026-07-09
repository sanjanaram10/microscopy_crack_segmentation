# app.py
import io
import os
import cv2
import torch
import tempfile
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from src.model import build_model
from src.preprocessing import preprocess_for_instance_seg
from src.config import Config
from typing import List

app = FastAPI()
cfg = Config()

# Load model once at startup
model = build_model(cfg.NUM_CLASSES)
checkpoint = torch.load("checkpoints/model_best.pth", map_location=cfg.DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(cfg.DEVICE)
model.eval()

COLORS = [
    (255, 0,   0),   (0, 255,   0),   (0,   0, 255),
    (255, 165, 0),   (128, 0, 128),   (0, 255, 255),
    (255, 20, 147),  (0, 128,   0),   (255, 215, 0),
    (0, 128, 128),   (255, 99,  71),  (30, 144, 255),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def calculate_crack_metrics(binary_mask):
    """Return area, perimeter, bounding box, and centroid for one mask."""
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
        "centroid":     [cx, cy],
    }


def mask_nms(masks, scores, iou_threshold=0.3):
    """
    Suppress duplicate masks using mask IoU.
    Keeps highest-confidence mask when two masks overlap above threshold.
    """
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
            union        = (masks[i] | masks[j]).sum()
            iou          = intersection / union if union > 0 else 0

            if iou > iou_threshold:
                suppressed.add(j)

    return kept


def merge_overlapping_masks(masks, scores, overlap_threshold=0.1):
    """
    Merge any remaining masks that still partially overlap.
    Uses overlap-ratio (intersection / smaller area) to catch partial overlaps.
    """
    if len(masks) == 0:
        return [], []

    used = set()
    merged_masks  = []
    merged_scores = []

    for i in range(len(masks)):
        if i in used:
            continue

        current_mask  = masks[i].copy()
        current_score = scores[i]

        for j in range(i + 1, len(masks)):
            if j in used:
                continue

            intersection  = (current_mask & masks[j]).sum()
            smaller_area  = min(current_mask.sum(), masks[j].sum())
            overlap_ratio = intersection / smaller_area if smaller_area > 0 else 0

            if overlap_ratio > overlap_threshold:
                current_mask  = (current_mask | masks[j]).astype(np.uint8)
                current_score = max(current_score, scores[j])
                used.add(j)

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        used.add(i)

    return merged_masks, merged_scores


# ── routes ────────────────────────────────────────────────────────────────────
import asyncio

async def process_single(file):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig_h, orig_w = img_bgr.shape[:2]

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        temp_path = tmp.name

    try:
        img_tensor = preprocess_for_instance_seg(temp_path)
        with torch.no_grad():
            predictions = model([img_tensor.to(cfg.DEVICE)])
    finally:
        os.unlink(temp_path)

    pred = predictions[0]
    result_img = img_bgr.copy()

    raw_masks = []
    raw_scores = []
    for score, mask in zip(pred['scores'], pred['masks']):
        if score < cfg.CONFIDENCE_THRESHOLD:
            continue
        binary = (mask.squeeze() > 0.5).cpu().numpy().astype(np.uint8)
        binary = cv2.resize(binary, (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST)
        raw_masks.append(binary)
        raw_scores.append(float(score))

    kept_indices = mask_nms(raw_masks, raw_scores, iou_threshold=0.1)
    nms_masks  = [raw_masks[i]  for i in kept_indices]
    nms_scores = [raw_scores[i] for i in kept_indices]

    final_masks, final_scores = merge_overlapping_masks(
        nms_masks, nms_scores, overlap_threshold=0.2
    )

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 165, 0), (128, 0, 128), (0, 255, 255),
        (255, 20, 147), (0, 128, 0), (255, 215, 0),
    ]

    cracks = []
    for crack_idx, (binary, score) in enumerate(zip(final_masks, final_scores)):
        metrics = calculate_crack_metrics(binary)
        metrics["score"] = round(score, 3)
        metrics["id"] = crack_idx
        color = colors[crack_idx % len(colors)]
        result_img[binary == 1] = color
        cracks.append(metrics)

    # Encode both images for frontend — this was missing
    _, buffer = cv2.imencode('.png', result_img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    _, orig_buffer = cv2.imencode('.png', img_bgr)
    orig_b64 = base64.b64encode(orig_buffer).decode('utf-8')

    # Return the result dict — this was missing
    return {
        "filename":     file.filename,
        "original_b64": orig_b64,
        "image_b64":    img_b64,
        "crack_count":  len(cracks),
        "cracks":       cracks
    }


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