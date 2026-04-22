import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def extract_detection_metrics(data: Dict[str, Any], gt_frames: Dict[str, List]) -> Tuple[List[int], List[float], List[int], List[float]]:
    counts = []
    total_conf = []
    absolute_errors = []
    over_detection_ratios = []

    for frame, detections in data.items():
        detection_count = len(detections)
        counts.append(detection_count)
        
        frame_gt = gt_frames.get(frame, [])
        dynamic_expected_count = len(frame_gt)

        for det in detections:
            total_conf.append(det.get("confidence", 0.0))

        abs_error = abs(detection_count - dynamic_expected_count)
        ratio = detection_count / dynamic_expected_count if dynamic_expected_count > 0 else 0.0

        absolute_errors.append(abs_error)
        over_detection_ratios.append(ratio)

    return counts, total_conf, absolute_errors, over_detection_ratios


def calculate_summary_stats(counts: List[int], total_conf: List[float], absolute_errors: List[int], over_detection_ratios: List[float]) -> Dict[str, float]:
    return {
        "avg_detections": float(np.mean(counts)) if counts else 0.0,
        "avg_confidence": float(np.mean(total_conf)) if total_conf else 0.0,
        "avg_abs_error": float(np.mean(absolute_errors)) if absolute_errors else 0.0,
        "avg_over_det_ratio": float(np.mean(over_detection_ratios)) if over_detection_ratios else 0.0
    }


def print_verbose_logs(data: Dict[str, Any], gt_frames: Dict[str, List]):
    """Prints frame-by-frame debug information."""
    print("\nFrame-by-Frame Breakdown:")
    for frame, detections in data.items():
        count = len(detections)
        dynamic_expected_count = len(gt_frames.get(frame, []))
        
        abs_error = abs(count - dynamic_expected_count)
        ratio = count / dynamic_expected_count if dynamic_expected_count > 0 else 0.0
        print(f"  {frame}: {count} det. (Expected: {dynamic_expected_count}) | abs. error = {abs_error} | over-det ratio = {ratio:.2f}")



def get_custom_metrics(data: Dict[str, Any], gt_frames: Dict[str, List], verbose: bool = False) -> Dict[str, float]:
    if verbose:
        print_verbose_logs(data, gt_frames)

    counts, confs, errors, ratios = extract_detection_metrics(data, gt_frames)
    return calculate_summary_stats(counts, confs, errors, ratios)

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    inter_x1, inter_y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    inter_x2, inter_y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def parse_ground_truth(gt_data: Dict) -> Tuple[Dict, int]:
    id_to_file = {str(img['image_id']): img['file_name'] for img in gt_data['images']}
    gt_frames = {}
    total_gt_count = 0
    
    for ann in gt_data['annotations']:
        if 'bbox_image' not in ann: 
            continue
            
        fname = id_to_file.get(str(ann['image_id']))
        if not fname: 
            continue
            
        attr = ann.get('attributes') or {}
        role, team = attr.get('role', ''), attr.get('team', '')
        
        if role == "player": cid = 0 if team == "left" else 1
        elif role == "goalkeeper": cid = 2 if team == "left" else 3
        else: cid = 4
            
        if fname not in gt_frames: 
            gt_frames[fname] = []
            
        b = ann['bbox_image']
        gt_frames[fname].append({
            'box': [b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']],
            'foot': [b['x_center'], b['y'] + b['h']],
            'class': cid
        })
        total_gt_count += 1
        
    return gt_frames, total_gt_count

def match_predictions(gt_frames: Dict, pred_data: Dict) -> Tuple:
    tp, fp, fn = 0, 0, 0
    foot_errors, all_detections = [], []
    y_true, y_pred = [], []

    for fname, preds in pred_data.items():
        gts = gt_frames.get(fname, [])
        matched_gt_indices = set()
        preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)

        for p in preds:
            p_box = p['bbox_image']
            best_iou, best_gt_idx = 0, -1

            for i, gt in enumerate(gts):
                if i in matched_gt_indices: continue
                iou = calculate_iou(p_box, gt['box'])
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, i

            if best_iou >= 0.5:
                tp += 1
                matched_gt_indices.add(best_gt_idx)
                all_detections.append((p['confidence'], 1))
                y_true.append(gts[best_gt_idx]['class'])
                y_pred.append(p['class_id'])
                
                # Calculate Foot-point Error
                p_foot = p['foot_point']
                gt_foot = gts[best_gt_idx]['foot']
                dist = np.sqrt((p_foot[0] - gt_foot[0])**2 + (p_foot[1] - gt_foot[1])**2)
                foot_errors.append(dist)
            else:
                fp += 1
                all_detections.append((p['confidence'], 0)) 
        
        fn += (len(gts) - len(matched_gt_indices))
        
    return tp, fp, fn, foot_errors, all_detections, y_true, y_pred


def calculate_and_print_metrics(tp: int, fp: int, fn: int, foot_errors: List[float], all_detections: List, total_gt_count: int, file_name: str, stats: Dict[str, float]):
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    rmse = np.sqrt(np.mean(np.square(foot_errors))) if foot_errors else 0

    # mAP@50 Calculation
    all_detections.sort(key=lambda x: x[0], reverse=True)
    tps = np.cumsum([x[1] for x in all_detections])
    fps = np.cumsum([1 - x[1] for x in all_detections])
    recalls = tps / total_gt_count if total_gt_count > 0 else np.zeros_like(tps)
    precisions = tps / (tps + fps)
    mAP50 = np.trapezoid(precisions, recalls) if len(recalls) > 0 else 0

    print(f"mAP@50:                       {mAP50:.2f}")
    print(f"Precision:                    {precision:.2f}")
    print(f"Recall:                       {recall:.2f}")
    print(f"F1-Score:                     {f1:.2f}")
    print(f"Foot-point RMSE:              {rmse:.2f} pixels")
    print(f"Average detections:           {stats['avg_detections']:.2f}")
    print(f"Average confidence:           {stats['avg_confidence']:.3f}")
    print(f"Average absolute error:       {stats['avg_abs_error']:.2f} objects")
    print(f"Average over-detection ratio: {stats['avg_over_det_ratio']:.2f}")

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], title: str):
    classes = ["Team A", "Team B", "GK A", "GK B", "Other"]
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title(title)
    plt.show()


def evaluate_game(gt_path: str, pred_path: str, model_name: str):
    with open(gt_path, 'r') as f: gt_data = json.load(f)
    with open(pred_path, 'r') as f: pred_data = json.load(f)

    gt_frames, total_gt_count = parse_ground_truth(gt_data)    
    tp, fp, fn, foot_errors, all_detections, y_true, y_pred = match_predictions(gt_frames, pred_data)
    file_basename = os.path.basename(pred_path)
    stats = get_custom_metrics(pred_data, gt_frames=gt_frames, verbose=False)

    print(f"\nPlayer Detection Metrics: {model_name}")
    calculate_and_print_metrics(tp, fp, fn, foot_errors, all_detections, total_gt_count, file_basename, stats)
    
    plot_confusion_matrix(y_true, y_pred, title=f"Confusion Matrix: {file_basename} ({model_name})")


if __name__ == "__main__":
    
    GT_FILE = "Test Values/SNGS-128/Labels-GameState.json"
    PRED_FILE = "Test Values/SNGS-128/SNGS_128_detection.json"
    MODEL_USED = "YOLOv8" 
    
    evaluate_game(GT_FILE, PRED_FILE, MODEL_USED)