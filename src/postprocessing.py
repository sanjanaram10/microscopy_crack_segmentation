import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def separate_touching_instances(binary_mask, min_distance=15):
    """
    Watershed separation for touching/overlapping instances.
    Returns integer label map (0=background, 1..N=instances)
    """
    distance = ndimage.distance_transform_edt(binary_mask)
    coords   = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary_mask
    )
    local_max          = np.zeros_like(distance, dtype=bool)
    local_max[tuple(coords.T)] = True
    markers            = ndimage.label(local_max)[0]
    return watershed(-distance, markers, mask=binary_mask)


def get_per_crack_stats(
    model, img_tensor,
    pixel_size_mm=None,
    confidence_threshold=0.5,
    min_area_px=50,
    device='cuda'
):
    model.eval()
    with torch.no_grad():
        predictions = model([img_tensor.to(device)])
    
    pred   = predictions[0]
    keep   = pred['scores'] > confidence_threshold
    masks  = pred['masks'][keep].squeeze(1).cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()
    boxes  = pred['boxes'][keep].cpu().numpy()
    
    scale = (pixel_size_mm ** 2) if pixel_size_mm else None
    
    results = []
    for i, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        binary = (mask > 0.5).astype(np.uint8)
        
        if binary.sum() < min_area_px:
            continue
        
        props = measure.regionprops(binary)
        if not props:
            continue
        p = props[0]
        
        minor = max(p.minor_axis_length, 1e-6)
        perim = max(p.perimeter, 1e-6)
        
        results.append({
            'crack_id':        i + 1,
            'confidence':      round(float(score), 3),
            'area_px':         int(p.area),
            'area_mm2':        round(p.area * scale, 4) if scale else None,
            'perimeter_px':    round(p.perimeter, 2),
            'eccentricity':    round(p.eccentricity, 3),
            'solidity':        round(p.solidity, 3),
            'aspect_ratio':    round(p.major_axis_length / minor, 3),
            'circularity':     round(4 * np.pi * p.area / perim**2, 3),
            'orientation_deg': round(np.degrees(p.orientation), 1),
            'centroid_x':      round(p.centroid[1], 1),
            'centroid_y':      round(p.centroid[0], 1),
            'bbox_x1':         round(float(box[0]), 1),
            'bbox_y1':         round(float(box[1]), 1),
            'bbox_x2':         round(float(box[2]), 1),
            'bbox_y2':         round(float(box[3]), 1),
        })
    
    return pd.DataFrame(results), masks