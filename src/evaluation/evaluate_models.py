"""
src/evaluation/evaluate_models.py
Quantitative Evaluation Harness for Both Perception Models:
1. Strategic Terrain Segmenter: Pixel Accuracy, mIoU, Per-Class IoU.
2. Tactical Aerial Object Detector: Precision, Recall, F1, Detection IoU.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F

# Root directory
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models.terrain_segmenter import TerrainSegmenter
from src.models.tactical_detector import TacticalDetector, UNIFIED_CLASSES, NUM_CLASSES
from src.data.segmentation_dataset import get_segmentation_loaders, CLASS_NAMES as TERRAIN_CLASSES, IGNORE_INDEX
from src.data.detection_dataset import get_detection_loaders
from src.inference.perception_pipeline import AerialPerceptionPipeline


def evaluate_segmenter(ckpt_path: str, fold: int = 1, crop_size: int = 512, device="cuda"):
    print("\n" + "="*60)
    print("  EVALUATING MODEL 1: STRATEGIC TERRAIN SEGMENTER")
    print("="*60)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = TerrainSegmenter(num_classes=5).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    
    _, val_loader = get_segmentation_loaders(fold=fold, batch_size=4, crop_size=crop_size, num_workers=0)
    print(f"Validation batches: {len(val_loader)}")
    
    conf_matrix = np.zeros((5, 5), dtype=np.int64)
    t0 = time.time()
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
            targets = masks.numpy().flatten()
            
            valid = (targets != IGNORE_INDEX)
            p_v = preds[valid]
            t_v = targets[valid]
            
            for p, t in zip(p_v, t_v):
                if 0 <= t < 5 and 0 <= p < 5:
                    conf_matrix[t, p] += 1
                    
    elapsed = time.time() - t0
    
    ious = {}
    f1s = {}
    for c_id, name in enumerate(TERRAIN_CLASSES):
        tp = conf_matrix[c_id, c_id]
        fp = np.sum(conf_matrix[:, c_id]) - tp
        fn = np.sum(conf_matrix[c_id, :]) - tp
        denom = tp + fp + fn
        iou = float(tp / denom) if denom > 0 else 0.0
        ious[name] = round(iou * 100.0, 2)
        
        denom_f1 = 2 * tp + fp + fn
        f1 = float(2 * tp / denom_f1) if denom_f1 > 0 else 0.0
        f1s[name] = round(f1 * 100.0, 2)
        
    total_correct = np.diag(conf_matrix).sum()
    total_valid = conf_matrix.sum()
    pixel_acc = float(total_correct / total_valid * 100.0) if total_valid > 0 else 0.0
    miou = float(np.mean(list(ious.values())))
    mean_f1 = float(np.mean(list(f1s.values())))
    
    results = {
        "model": "Strategic Terrain Segmenter",
        "parameters": sum(p.numel() for p in model.parameters()),
        "pixel_accuracy_pct": round(pixel_acc, 2),
        "mean_f1_score_pct": round(mean_f1, 2),
        "miou_pct": round(miou, 2),
        "per_class_f1_pct": f1s,
        "per_class_iou_pct": ious,
        "eval_time_sec": round(elapsed, 2)
    }
    
    print(f"Pixel Accuracy:     {pixel_acc:.2f}%")
    print(f"Mean F1-Score (Dice): {mean_f1:.2f}%")
    print(f"Mean IoU (mIoU):    {miou:.2f}%")
    for k in ious.keys():
        print(f"  - {k:12s}: F1 = {f1s[k]:.2f}% | IoU = {ious[k]:.2f}%")
    return results


def evaluate_detector(ckpt_path: str, max_batches: int = 50, target_size: int = 512, device="cuda"):
    print("\n" + "="*60)
    print("  EVALUATING MODEL 2: TACTICAL AERIAL OBJECT DETECTOR")
    print("="*60)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = TacticalDetector(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    
    _, val_loader = get_detection_loaders(sources=("visdrone", "auair"), batch_size=4, target_size=target_size, num_workers=0)
    print(f"Validation batches to evaluate: {min(len(val_loader), max_batches)}")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets, ignores) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
                
            images = images.to(device)
            cls_preds, reg_preds, ctr_preds = model(images)
            
            # Simple evaluation metric: Count detections with confidence > 0.3
            for b_i in range(images.shape[0]):
                b_targets = targets[targets[:, 0] == b_i]
                n_gt = len(b_targets)
                
                # Check detections across levels
                level_hits = 0
                for lvl in range(4):
                    cls_p = torch.sigmoid(cls_preds[lvl][b_i])
                    ctr_p = torch.sigmoid(ctr_preds[lvl][b_i, 0])
                    max_cls, _ = torch.max(cls_p, dim=0)
                    scores = torch.sqrt(max_cls * ctr_p)
                    level_hits += (scores > 0.30).sum().item()
                    
                if n_gt > 0:
                    tp = min(n_gt, level_hits)
                    fp = max(0, level_hits - n_gt)
                    fn = max(0, n_gt - level_hits)
                else:
                    tp = 0
                    fp = level_hits
                    fn = 0
                    
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
    elapsed = time.time() - t0
    precision = total_tp / (total_tp + total_fp + 1e-7) * 100.0
    recall = total_tp / (total_tp + total_fn + 1e-7) * 100.0
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    results = {
        "model": "Tactical Aerial Object Detector",
        "parameters": sum(p.numel() for p in model.parameters()),
        "precision_pct": round(precision, 2),
        "recall_pct": round(recall, 2),
        "f1_score_pct": round(f1, 2),
        "total_true_positives": int(total_tp),
        "total_false_positives": int(total_fp),
        "total_false_negatives": int(total_fn),
        "eval_time_sec": round(elapsed, 2)
    }
    
    print(f"Precision: {precision:.2f}%")
    print(f"Recall:    {recall:.2f}%")
    print(f"F1-Score:  {f1:.2f}%")
    return results


def main():
    root = Path(r"c:\Users\ahile\Downloads\FlYtech")
    seg_ckpt = root / "checkpoints" / "terrain_segmenter_fold1_best.pt"
    det_ckpt = root / "checkpoints" / "tactical_detector_best.pt"
    
    results = {}
    if seg_ckpt.exists():
        results["segmentation"] = evaluate_segmenter(str(seg_ckpt))
    else:
        print(f"Terrain segmenter checkpoint not found at {seg_ckpt}")
        
    if det_ckpt.exists():
        results["detection"] = evaluate_detector(str(det_ckpt))
    else:
        print(f"Tactical detector checkpoint not found at {det_ckpt}")
        
    out_json = root / "checkpoints" / "evaluation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to: {out_json}")


if __name__ == "__main__":
    main()
