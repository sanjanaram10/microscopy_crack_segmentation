import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def _mask_bbox(binary_mask):
    ys, xs = np.where(binary_mask > 0)
    if xs.size == 0:
        return [0, 0, 0, 0]

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def merge_overlapping_instances(masks, scores=None):
    """
    Merge any instance masks that overlap into a single crack component.

    The merged crack keeps the union of all pixels from the overlap group and
    uses the largest member as the representative for ordering and score.
    """
    if masks is None or len(masks) == 0:
        if scores is None:
            return [], []
        return [], [], []

    binary_masks = []
    for mask in masks:
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        binary_masks.append(np.asarray(mask).astype(bool))

    scores_arr = None if scores is None else np.asarray(scores)
    areas = np.array([mask.sum() for mask in binary_masks])

    parents = list(range(len(binary_masks)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if areas[left_root] < areas[right_root]:
                left_root, right_root = right_root, left_root
            parents[right_root] = left_root

    for left in range(len(binary_masks)):
        for right in range(left + 1, len(binary_masks)):
            if np.any(binary_masks[left] & binary_masks[right]):
                union(left, right)

    groups = {}
    for index in range(len(binary_masks)):
        root = find(index)
        groups.setdefault(root, []).append(index)

    merged_masks = []
    merged_scores = [] if scores_arr is not None else None
    merged_bboxes = []

    for group in groups.values():
        group = sorted(group)
        union_mask = np.any([binary_masks[index] for index in group], axis=0).astype(np.uint8)
        representative = max(
            group,
            key=lambda index: (
                areas[index],
                float(scores_arr[index]) if scores_arr is not None else 0.0,
                -index,
            ),
        )

        merged_masks.append(union_mask)
        merged_bboxes.append(_mask_bbox(union_mask))
        if merged_scores is not None:
            merged_scores.append(float(scores_arr[representative]))

    order = sorted(
        range(len(merged_masks)),
        key=lambda index: (-int(merged_masks[index].sum()), merged_bboxes[index][0], merged_bboxes[index][1]),
    )

    merged_masks = [merged_masks[index] for index in order]
    merged_bboxes = [merged_bboxes[index] for index in order]
    if merged_scores is not None:
        merged_scores = [merged_scores[index] for index in order]
        return merged_masks, merged_scores, merged_bboxes

    return merged_masks, merged_bboxes

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

    merged = merge_overlapping_instances(masks, scores)
    merged_masks, merged_scores, merged_bboxes = merged
    
    scale = (pixel_size_mm ** 2) if pixel_size_mm else None
    
    results = []
    for i, (binary, score, bbox) in enumerate(zip(merged_masks, merged_scores, merged_bboxes)):
        binary = binary.astype(np.uint8)
        
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
            'bbox_x1':         round(float(bbox[0]), 1),
            'bbox_y1':         round(float(bbox[1]), 1),
            'bbox_x2':         round(float(bbox[0] + bbox[2]), 1),
            'bbox_y2':         round(float(bbox[1] + bbox[3]), 1),
        })
    
    return pd.DataFrame(results), merged_masks