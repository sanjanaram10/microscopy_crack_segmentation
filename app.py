# app.py
import io
import os
import cv2
import torch
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import base64
from src.model import build_model
from src.preprocessing import preprocess_for_instance_seg
from src.config import Config

app = FastAPI()
cfg = Config()

model = build_model(cfg.NUM_CLASSES)
checkpoint = torch.load("checkpoints/model_best.pth", map_location=cfg.DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(cfg.DEVICE)
model.eval()

def calculate_crack_metrics(binary_mask):
    """Calculate area and perimeter of a binary mask in pixels."""
    area = int(binary_mask.sum())

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)

    # Bounding box for hover detection
    if contours:
        x, y, w, h = cv2.boundingRect(
            np.concatenate(contours)
        )
    else:
        x, y, w, h = 0, 0, 0, 0

    # Centroid for label placement
    if area > 0:
        M = cv2.moments(binary_mask)
        cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else x + w // 2
        cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else y + h // 2
    else:
        cx, cy = x + w // 2, y + h // 2

    return {
        "area_px":    area,
        "perimeter_px": round(perimeter, 1),
        "bbox":       [x, y, w, h],   # for hover hit detection
        "centroid":   [cx, cy]         # for tooltip placement
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
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
    cracks = []

    # Assign a distinct color per crack for hover matching
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 165, 0), (128, 0, 128), (0, 255, 255),
        (255, 20, 147), (0, 128, 0), (255, 215, 0),
    ]

    crack_idx = 0
    for score, mask in zip(pred['scores'], pred['masks']):
        if score < cfg.CONFIDENCE_THRESHOLD:
            continue

        binary = (mask.squeeze() > 0.5).cpu().numpy().astype(np.uint8)
        binary = cv2.resize(binary, (orig_w, orig_h),
                           interpolation=cv2.INTER_NEAREST)

        metrics = calculate_crack_metrics(binary)
        metrics["score"] = round(float(score), 3)
        metrics["id"] = crack_idx

        # Draw mask with unique color
        color = colors[crack_idx % len(colors)]
        result_img[binary == 1] = color

        cracks.append(metrics)
        crack_idx += 1

    # Encode annotated image as base64 to return alongside JSON
    _, buffer = cv2.imencode('.png', result_img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        "image_b64": img_b64,
        "crack_count": len(cracks),
        "cracks": cracks
    })


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")