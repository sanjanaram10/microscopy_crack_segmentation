# app.py
import io
import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from src.model import build_model
from src.preprocessing import preprocess_for_instance_seg
from src.config import Config
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
cfg = Config()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

# Load model once at startup
model = build_model(cfg.NUM_CLASSES)
checkpoint = torch.load("checkpoints/model_best.pth", map_location=cfg.DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(cfg.DEVICE)
model.eval()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save temporarily and preprocess
    temp_path = f"/tmp/{file.filename}"
    cv2.imwrite(temp_path, img_bgr)
    img_tensor = preprocess_for_instance_seg(temp_path)
    
    # Run inference
    with torch.no_grad():
        predictions = model([img_tensor.to(cfg.DEVICE)])
    
    pred = predictions[0]
    orig_h, orig_w = img_bgr.shape[:2]
    
    # Draw results on image
    result_img = img_bgr.copy()
    crack_count = 0
    
    for score, mask in zip(pred['scores'], pred['masks']):
        if score < cfg.CONFIDENCE_THRESHOLD:
            continue
        crack_count += 1
        
        # Resize mask to original image size
        binary = (mask.squeeze() > 0.5).cpu().numpy().astype(np.uint8)
        binary = cv2.resize(binary, (orig_w, orig_h), 
                           interpolation=cv2.INTER_NEAREST)
        
        # Overlay mask in red
        result_img[binary == 1] = [0, 0, 255]
    
    # Return annotated image
    _, buffer = cv2.imencode('.png', result_img)
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={
            "X-Crack-Count": str(crack_count),
            "X-Confidence-Threshold": str(cfg.CONFIDENCE_THRESHOLD)
        }
    )

@app.get("/health")
def health():
    return {"status": "ok"}