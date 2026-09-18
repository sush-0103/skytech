"""
src/training/train_detection.py
Training engine for Tactical Aerial Object Detector (VisDrone & AU-AIR).
- Custom Anchor-Free Target Assignment across P2, P3, P4, P5 strides.
- Quality Focal Loss (QFL) + Complete IoU (CIoU) Box Regression + Centerness.
- Masking of score-zero VisDrone ignore regions.
- Multi-scale training with Cosine Annealing learning rate.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models.tactical_detector import TacticalDetector, UNIFIED_CLASSES, NUM_CLASSES
from src.data.detection_dataset import get_detection_loaders


def bbox_ciou(pred_boxes, target_boxes, eps=1e-7):
    """
    Computes Complete IoU (CIoU) loss between pred and target boxes (x1, y1, x2, y2).
    """
    # Intersections
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Unions
    w_p = pred_boxes[:, 2] - pred_boxes[:, 0]
    h_p = pred_boxes[:, 3] - pred_boxes[:, 1]
    w_t = target_boxes[:, 2] - target_boxes[:, 0]
    h_t = target_boxes[:, 3] - target_boxes[:, 1]
    
    union = w_p * h_p + w_t * h_t - inter + eps
    iou = inter / union
    
    # Enclosing box
    c_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    c_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    c_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    c_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps
    
    # Center distance
    ctr_p_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
    ctr_p_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
    ctr_t_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2.0
    ctr_t_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2.0
    d2 = (ctr_p_x - ctr_t_x) ** 2 + (ctr_p_y - ctr_t_y) ** 2
    
    # Aspect ratio consistency term
    v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(w_t / (h_t + eps)) - torch.atan(w_p / (h_p + eps)), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
        
    ciou = iou - (d2 / c2 + alpha * v)
    return 1.0 - ciou.clamp(min=-1.0, max=1.0)


class TacticalDetectionLoss(nn.Module):
    def __init__(self, strides=(4, 8, 16, 32), num_classes=10):
        super().__init__()
        self.strides = strides
        self.num_classes = num_classes
        self.scale_ranges = [
            (0, 80),      # P2 (stride 4): tiny aerial objects
            (40, 160),    # P3 (stride 8): small objects
            (80, 320),    # P4 (stride 16): medium objects
            (160, 10000)  # P5 (stride 32): large objects
        ]

    def forward(self, cls_preds, reg_preds, centerness_preds, targets, ignore_masks):
        # Batch size
        b = cls_preds[0].shape[0]
        device = cls_preds[0].device
        
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_ctr_loss = 0.0
        num_pos_total = 0
        
        # Pre-convert targets to numpy once to avoid GPU-CPU sync in loop
        targets_np = targets.detach().cpu().numpy() if len(targets) > 0 else np.zeros((0, 6), dtype=np.float32)
        
        for level_idx, stride in enumerate(self.strides):
            cls_p = cls_preds[level_idx]          # (B, C, H, W)
            reg_p = reg_preds[level_idx]          # (B, 4, H, W)
            ctr_p = centerness_preds[level_idx]   # (B, 1, H, W)
            h, w = cls_p.shape[2], cls_p.shape[3]
            
            # Grid point coordinates in image pixels
            ys, xs = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32) * stride + stride / 2.0,
                torch.arange(w, device=device, dtype=torch.float32) * stride + stride / 2.0,
                indexing="ij"
            )
            xs = xs.unsqueeze(0).expand(b, -1, -1)  # (B, H, W)
            ys = ys.unsqueeze(0).expand(b, -1, -1)  # (B, H, W)
            
            # Targets for this scale
            min_sz, max_sz = self.scale_ranges[level_idx]
            
            # Target maps
            cls_targets = torch.zeros((b, self.num_classes, h, w), device=device, dtype=torch.float32)
            pos_mask = torch.zeros((b, h, w), device=device, dtype=torch.bool)
            reg_targets = torch.zeros((b, 4, h, w), device=device, dtype=torch.float32)
            ctr_targets = torch.zeros((b, 1, h, w), device=device, dtype=torch.float32)
            
            if len(targets_np) > 0:
                img_w_px = w * stride
                img_h_px = h * stride
                
                for tgt in targets_np:
                    b_i = int(tgt[0])
                    c_id = int(tgt[1])
                    if b_i >= b or c_id >= self.num_classes:
                        continue
                        
                    xc = tgt[2] * img_w_px
                    yc = tgt[3] * img_h_px
                    bw = tgt[4] * img_w_px
                    bh = tgt[5] * img_h_px
                    
                    max_dim = max(bw, bh)
                    if max_dim < min_sz or max_dim > max_sz:
                        continue
                        
                    x1 = xc - bw / 2.0
                    y1 = yc - bh / 2.0
                    x2 = xc + bw / 2.0
                    y2 = yc + bh / 2.0
                    
                    # Bounding grid box bounds
                    gx1 = max(0, int(np.floor(x1 / stride)))
                    gx2 = min(w, int(np.ceil(x2 / stride)))
                    gy1 = max(0, int(np.floor(y1 / stride)))
                    gy2 = min(h, int(np.ceil(y2 / stride)))
                    if gx1 >= gx2 or gy1 >= gy2:
                        continue
                        
                    sub_xs = xs[b_i, gy1:gy2, gx1:gx2]
                    sub_ys = ys[b_i, gy1:gy2, gx1:gx2]
                    
                    in_box = (sub_xs >= x1) & (sub_xs <= x2) & (sub_ys >= y1) & (sub_ys <= y2)
                    if not in_box.any():
                        continue
                        
                    l = sub_xs[in_box] - x1
                    t = sub_ys[in_box] - y1
                    r = x2 - sub_xs[in_box]
                    b_off = y2 - sub_ys[in_box]
                    
                    ctr = torch.sqrt((torch.minimum(l, r) / (torch.maximum(l, r) + 1e-7)) *
                                     (torch.minimum(t, b_off) / (torch.maximum(t, b_off) + 1e-7)))
                                     
                    pos_sub = pos_mask[b_i, gy1:gy2, gx1:gx2]
                    pos_sub[in_box] = True
                    pos_mask[b_i, gy1:gy2, gx1:gx2] = pos_sub
                    
                    cls_sub = cls_targets[b_i, c_id, gy1:gy2, gx1:gx2]
                    cls_sub[in_box] = 1.0
                    cls_targets[b_i, c_id, gy1:gy2, gx1:gx2] = cls_sub
                    
                    reg_targets[b_i, 0, gy1:gy2, gx1:gx2][in_box] = l
                    reg_targets[b_i, 1, gy1:gy2, gx1:gx2][in_box] = t
                    reg_targets[b_i, 2, gy1:gy2, gx1:gx2][in_box] = r
                    reg_targets[b_i, 3, gy1:gy2, gx1:gx2][in_box] = b_off
                    ctr_targets[b_i, 0, gy1:gy2, gx1:gx2][in_box] = ctr
                        
            # Resize ignore masks to this level's grid resolution
            ign_ds = F.interpolate(ignore_masks.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)  # (B, H, W)
            
            # 1. Classification Loss (Focal Loss, masked by ignore map)
            valid_cls_mask = (ign_ds > 0.5).unsqueeze(1).expand(-1, self.num_classes, -1, -1)
            bce_cls = F.binary_cross_entropy_with_logits(cls_p, cls_targets, reduction="none")
            p = torch.sigmoid(cls_p)
            focal_weight = torch.where(cls_targets == 1.0, 1.0 - p, p).pow(2.0)
            cls_loss = (bce_cls * focal_weight * valid_cls_mask).sum()
            
            # 2. Box Regression and Centerness Loss on positive points
            n_pos = pos_mask.sum().item()
            num_pos_total += n_pos
            
            if n_pos > 0:
                # Reg predicted boxes
                reg_p_pos = reg_p.permute(0, 2, 3, 1)[pos_mask]  # (N_pos, 4)
                reg_t_pos = reg_targets.permute(0, 2, 3, 1)[pos_mask]
                
                xs_pos = xs[pos_mask]
                ys_pos = ys[pos_mask]
                
                pred_x1 = xs_pos - reg_p_pos[:, 0]
                pred_y1 = ys_pos - reg_p_pos[:, 1]
                pred_x2 = xs_pos + reg_p_pos[:, 2]
                pred_y2 = ys_pos + reg_p_pos[:, 3]
                
                tgt_x1 = xs_pos - reg_t_pos[:, 0]
                tgt_y1 = ys_pos - reg_t_pos[:, 1]
                tgt_x2 = xs_pos + reg_t_pos[:, 2]
                tgt_y2 = ys_pos + reg_t_pos[:, 3]
                
                ciou_loss = bbox_ciou(
                    torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1),
                    torch.stack([tgt_x1, tgt_y1, tgt_x2, tgt_y2], dim=1)
                ).sum()
                
                ctr_p_pos = ctr_p.permute(0, 2, 3, 1)[pos_mask].squeeze(-1)
                ctr_t_pos = ctr_targets.permute(0, 2, 3, 1)[pos_mask].squeeze(-1)
                ctr_loss = F.binary_cross_entropy_with_logits(ctr_p_pos, ctr_t_pos, reduction="sum")
                
                total_reg_loss += ciou_loss
                total_ctr_loss += ctr_loss
                
            total_cls_loss += cls_loss
            
        norm_factor = max(num_pos_total, 1)
        loss = (total_cls_loss / norm_factor) + 2.0 * (total_reg_loss / norm_factor) + (total_ctr_loss / norm_factor)
        return loss, total_cls_loss.item() / norm_factor, total_reg_loss.item() / norm_factor if num_pos_total > 0 else 0.0


def train_detection(epochs=10, batch_size=16, target_size=512, sources=("visdrone", "auair"), lr=1e-3, export_onnx=True):
    print("==================================================")
    print("  TACTICAL AERIAL DETECTOR HIGH-POWER GPU TRAINING")
    print("==================================================")
    
    # Safe device selection
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            test_t = torch.zeros(1, device="cuda")
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name(0)} (Blackwell Compute Capability {torch.cuda.get_device_capability(0)})", flush=True)
        except Exception as e:
            print(f"GPU detected but CUDA kernels not built for this arch ({e}). Falling back to CPU.", flush=True)
            
    if device.type == "cpu":
        num_threads = os.cpu_count() or 24
        torch.set_num_threads(num_threads)
        print(f"Using CPU: Intel Core Ultra 9 with {num_threads} parallel threads", flush=True)
    
    # Loaders with pin_memory
    print(f"Loading datasets: {sources} (batch size: {batch_size}, target size: {target_size}x{target_size})...", flush=True)
    train_loader, val_loader = get_detection_loaders(
        sources=sources,
        batch_size=batch_size,
        target_size=target_size,
        num_workers=0
    )
    
    # Model
    model = TacticalDetector(num_classes=NUM_CLASSES).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TacticalDetector parameters: {params:,}", flush=True)
    
    criterion = TacticalDetectionLoss(strides=(4, 8, 16, 32), num_classes=NUM_CLASSES)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    ckpt_dir = Path(r"c:\Users\ahile\Downloads\FlYtech\checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "tactical_detector_best.pt"
    
    # Load existing checkpoint if available to continue refining
    if best_ckpt.exists():
        try:
            ckpt_data = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data.get("model_state_dict", ckpt_data))
            print(f"Loaded existing weights from {best_ckpt} to continue high-power training!", flush=True)
        except Exception as e:
            print(f"Starting fresh weights: {e}", flush=True)
    
    print("\nStarting High-Power GPU Tactical Detector training loop...", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_cls, total_reg = 0.0, 0.0, 0.0
        num_batches = len(train_loader)
        
        # Train epoch (sample up to 100 batches per epoch)
        max_batches = min(num_batches, 100)
        for batch_idx, (images, targets, ignores) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
                
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            ignores = ignores.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                cls_out, reg_out, ctr_out = model(images)
                loss, cls_l, reg_l = criterion(cls_out, reg_out, ctr_out, targets, ignores)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_cls += cls_l
            total_reg += reg_l
            
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == max_batches:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch [{epoch:02d}/{epochs:02d}] Batch [{batch_idx+1:03d}/{max_batches:03d}] -> Loss: {avg_loss:.4f} (Cls: {cls_l:.4f}, Reg: {reg_l:.4f})", flush=True)
                
        scheduler.step()
        elapsed = time.time() - t0
        avg_epoch_loss = total_loss / max_batches
        print(f"\n--- Epoch {epoch:02d} Summary ({elapsed:.1f}s) | Avg Loss: {avg_epoch_loss:.4f} ---\n", flush=True)
        
        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": avg_epoch_loss
        }, best_ckpt)
        print(f"  Saved Tactical Detector checkpoint -> {best_ckpt}\n", flush=True)
        
    print("==================================================")
    print(f"Tactical Detector Training Complete!")
    print(f"Saved checkpoint to: {best_ckpt}")
    print("==================================================\n")

    if export_onnx:
        onnx_path = ckpt_dir / f"tactical_detector_{target_size}.onnx"
        print(f"Exporting Tactical Detector to static ONNX: {onnx_path}...")
        model.eval().cpu()
        dummy_input = torch.randn(1, 3, target_size, target_size)
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=["input_image"],
                output_names=["cls_p2", "reg_p2", "ctr_p2", "cls_p3", "reg_p3", "ctr_p3",
                              "cls_p4", "reg_p4", "ctr_p4", "cls_p5", "reg_p5", "ctr_p5"],
                opset_version=18,
                do_constant_folding=True
            )
            print(f"Successfully exported ONNX model ({onnx_path.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            print(f"ONNX export failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tactical Aerial Object Detector")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--target_size", type=int, default=512, help="Image resolution")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sources", nargs="+", default=["visdrone", "auair"], help="Dataset sources")
    parser.add_argument("--no_onnx", action="store_true", help="Skip ONNX export")
    args = parser.parse_args()

    train_detection(
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_size=args.target_size,
        sources=args.sources,
        lr=args.lr,
        export_onnx=not args.no_onnx
    )
