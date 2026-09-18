"""
src/inference/perception_pipeline.py
Integrated Aerial Perception Pipeline for Autonomous Drone Operations.
- Model 1: Strategic Terrain Segmenter (Water, Land, Road, Building, Vegetation, Landing Zones)
- Model 2: Tactical Aerial Object Detector (Vehicles, Pedestrians, Drones, Craft)
- Strict Non-Learned Flight Advisory: Outputs risk costmaps and detected obstacles
  to SITL / Mission Planner without direct actuator control authority.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

# Root directory
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models.terrain_segmenter import TerrainSegmenter
from src.models.tactical_detector import TacticalDetector, UNIFIED_CLASSES, NUM_CLASSES
from src.data.segmentation_dataset import CLASS_NAMES as TERRAIN_CLASSES

# Color maps for visualization
TERRAIN_COLORS = {
    0: (132, 41, 246),   # Land (Purple)
    1: (226, 169, 41),   # Water (Blue-orange/cyan)
    2: (60, 16, 152),    # Building (Dark Purple)
    3: (254, 221, 58),   # Vegetation (Yellow)
    4: (110, 193, 228),  # Road (Light Blue)
    255: (0, 0, 0)       # Ignore / Unlabeled
}

# Terrain flight risk cost (0.0 = safest, 1.0 = hazardous / lethal)
TERRAIN_RISK_COSTS = {
    0: 0.20,  # Land: standard open terrain
    1: 1.00,  # Water: catastrophic loss hazard for multirotors
    2: 0.90,  # Building: structural obstacle
    3: 0.40,  # Vegetation: soft obstacle / canopy
    4: 0.05,  # Road: emergency paved landing strip
    255: 0.50 # Unknown
}


class AerialPerceptionPipeline:
    def __init__(
        self,
        segmenter_ckpt: str = None,
        detector_ckpt: str = None,
        device: str = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[AerialPerceptionPipeline] Initializing on device: {self.device}")
        
        # 1. Load Strategic Terrain Segmenter
        self.segmenter = TerrainSegmenter(num_classes=5).to(self.device)
        if segmenter_ckpt and Path(segmenter_ckpt).exists():
            ckpt = torch.load(segmenter_ckpt, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.segmenter.load_state_dict(state)
            print(f"  Loaded Terrain Segmenter from {segmenter_ckpt}")
        self.segmenter.eval()
        
        # 2. Load Tactical Aerial Detector
        self.detector = TacticalDetector(num_classes=NUM_CLASSES).to(self.device)
        if detector_ckpt and Path(detector_ckpt).exists():
            ckpt = torch.load(detector_ckpt, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.detector.load_state_dict(state)
            print(f"  Loaded Tactical Detector from {detector_ckpt}")
        self.detector.eval()
        
    @torch.no_grad()
    def segment_terrain(self, pil_img: Image.Image, target_size: int = 512):
        """
        Runs Strategic Terrain Segmentation and produces risk costmap.
        """
        w, h = pil_img.size
        resized = pil_img.resize((target_size, target_size), Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (arr - mean) / std
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # Forward pass (eval mode returns semantic logits)
        logits = self.segmenter(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)  # (5, H, W)
        pred_classes = torch.argmax(probs, dim=0).cpu().numpy()  # (H, W)
        
        # Resize class map back to original image size
        pred_pil = Image.fromarray(pred_classes.astype(np.uint8)).resize((w, h), Image.NEAREST)
        full_pred = np.array(pred_pil)
        
        # Calculate landing zone safety and costmap
        costmap = np.zeros((h, w), dtype=np.float32)
        for class_id, cost in TERRAIN_RISK_COSTS.items():
            costmap[full_pred == class_id] = cost
            
        # Class distribution statistics
        total_px = h * w
        stats = {}
        for c_id, name in enumerate(TERRAIN_CLASSES):
            pct = (full_pred == c_id).sum() / total_px * 100.0
            stats[name] = round(pct, 2)
            
        return {
            "pred_map": full_pred,
            "costmap": costmap,
            "class_distribution": stats,
            "recommended_landing_zone": "Road / Flat Land" if (stats.get("Road", 0) > 5 or stats.get("Land", 0) > 20) else "Hazard: High Water / Structure Density"
        }

    @torch.no_grad()
    def detect_objects(
        self,
        pil_img: Image.Image,
        target_size: int = 512,
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.45
    ):
        """
        Runs Tactical Aerial Detector and decodes bounding boxes across P2-P5 scales.
        """
        w_orig, h_orig = pil_img.size
        
        # Letterbox to target_size
        scale = min(target_size / w_orig, target_size / h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
        letterboxed = Image.new("RGB", (target_size, target_size), (114, 114, 114))
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        letterboxed.paste(resized, (pad_x, pad_y))
        
        arr = np.array(letterboxed, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        cls_preds, reg_preds, ctr_preds = self.detector(tensor)
        strides = [4, 8, 16, 32]
        
        candidates = []
        for level_idx, stride in enumerate(strides):
            cls_p = torch.sigmoid(cls_preds[level_idx][0])       # (C, H, W)
            reg_p = reg_preds[level_idx][0]                      # (4, H, W)
            ctr_p = torch.sigmoid(ctr_preds[level_idx][0, 0])    # (H, W)
            
            h_grid, w_grid = cls_p.shape[1], cls_p.shape[2]
            
            # Grid points in letterbox pixels
            ys, xs = torch.meshgrid(
                torch.arange(h_grid, device=self.device, dtype=torch.float32) * stride + stride / 2.0,
                torch.arange(w_grid, device=self.device, dtype=torch.float32) * stride + stride / 2.0,
                indexing="ij"
            )
            
            # Combined score: sqrt(cls * ctr)
            max_cls_score, class_ids = torch.max(cls_p, dim=0)  # (H, W)
            combined_scores = torch.sqrt(max_cls_score * ctr_p)
            
            mask = combined_scores > conf_thresh
            if not mask.any():
                continue
                
            xs_sel = xs[mask]
            ys_sel = ys[mask]
            scores_sel = combined_scores[mask]
            classes_sel = class_ids[mask]
            
            # l, t, r, b
            l = reg_p[0][mask]
            t = reg_p[1][mask]
            r = reg_p[2][mask]
            b = reg_p[3][mask]
            
            bx1 = xs_sel - l
            by1 = ys_sel - t
            bx2 = xs_sel + r
            by2 = ys_sel + b
            
            # Map back from letterbox coords to original image coords
            orig_x1 = (bx1 - pad_x) / scale
            orig_y1 = (by1 - pad_y) / scale
            orig_x2 = (bx2 - pad_x) / scale
            orig_y2 = (by2 - pad_y) / scale
            
            for i in range(len(scores_sel)):
                candidates.append({
                    "box": [
                        float(orig_x1[i].item()),
                        float(orig_y1[i].item()),
                        float(orig_x2[i].item()),
                        float(orig_y2[i].item())
                    ],
                    "score": float(scores_sel[i].item()),
                    "class_id": int(classes_sel[i].item()),
                    "class_name": UNIFIED_CLASSES[int(classes_sel[i].item())]
                })
                
        # Non-Maximum Suppression (NMS)
        if not candidates:
            return []
            
        boxes_t = torch.tensor([c["box"] for c in candidates], dtype=torch.float32)
        scores_t = torch.tensor([c["score"] for c in candidates], dtype=torch.float32)
        
        keep_indices = self._nms(boxes_t, scores_t, nms_thresh)
        final_dets = [candidates[i] for i in keep_indices]
        return final_dets

    def _nms(self, boxes, scores, thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
            inds = (ovr <= thresh).nonzero().squeeze()
            if inds.numel() == 0:
                break
            order = order[inds + 1]
            if order.ndim == 0:
                order = order.unsqueeze(0)
        return keep

    def render_tactical_overlay(
        self,
        pil_img: Image.Image,
        terrain_result: dict,
        detections: list,
        alpha: float = 0.35
    ) -> Image.Image:
        """
        Creates an annotated visual contact sheet combining semantic terrain
        coloring, risk overlay, and tactical detection bounding boxes.
        """
        w, h = pil_img.size
        # 1. Colorize terrain segmentation
        pred_map = terrain_result["pred_map"]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for c_id, rgb in TERRAIN_COLORS.items():
            color_mask[pred_map == c_id] = rgb
            
        color_mask_img = Image.fromarray(color_mask)
        blended = Image.blend(pil_img.convert("RGB"), color_mask_img, alpha=alpha)
        
        # 2. Draw tactical bounding boxes
        draw = ImageDraw.Draw(blended)
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class_name"]
            score = det["score"]
            
            # Box border
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 128), width=2)
            label = f"{cls_name} {score:.2f}"
            draw.rectangle([x1, max(0, y1 - 16), x1 + len(label) * 7 + 4, max(16, y1)], fill=(0, 0, 0))
            draw.text((x1 + 2, max(0, y1 - 15)), label, fill=(255, 255, 255))
            
        return blended
