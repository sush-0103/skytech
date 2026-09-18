"""
src/training/train_segmentation.py
High-F1 Strategic Terrain Segmentation Training Engine on Dubai Aerial Dataset.
- Custom Compound Loss optimized for maximum F1 (Dice): 0.8 * CE + 1.6 * SoftDice + 0.3 * BoundaryBCE.
- Rebalanced class weights targeting minority classes (Road, Building, Vegetation).
- Full GPU acceleration with PyTorch AMP FP16, TensorFloat-32, and cuDNN benchmark.
- Tracks per-class F1 scores and mIoU every epoch.
- Automatic ONNX export upon convergence.
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

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models.terrain_segmenter import TerrainSegmenter
from src.data.segmentation_dataset import get_segmentation_loaders, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX

# Rebalanced inverse-frequency weights prioritizing minority classes for higher F1
# Land, Water, Building, Vegetation, Road
CLASS_WEIGHTS = [0.60, 1.20, 2.20, 2.00, 2.60]


class SoftDiceLoss(nn.Module):
    """Multi-class Soft Dice Loss directly optimizing the F1-Score."""
    def __init__(self, num_classes=5, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index)
        
        clamped_targets = targets.clone()
        clamped_targets[~valid_mask] = 0
        
        one_hot = F.one_hot(clamped_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        probs = probs * valid_mask.unsqueeze(1)
        one_hot = one_hot * valid_mask.unsqueeze(1)
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        cardinality = torch.sum(probs * probs + one_hot * one_hot, dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        return 1.0 - torch.mean(dice_score)


def extract_sobel_boundaries(masks, stride=4):
    b, h, w = masks.shape
    ds_h, ds_w = h // stride, w // stride
    
    downsampled = F.interpolate(
        masks.unsqueeze(1).float(),
        size=(ds_h, ds_w),
        mode="nearest"
    ).squeeze(1).long()
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=masks.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=masks.device).view(1, 1, 3, 3)
    
    clean_ds = downsampled.clone()
    clean_ds[clean_ds == IGNORE_INDEX] = 0
    clean_ds = clean_ds.unsqueeze(1).float()
    
    grad_x = F.conv2d(clean_ds, sobel_x, padding=1)
    grad_y = F.conv2d(clean_ds, sobel_y, padding=1)
    
    edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    boundary = (edge_magnitude > 0.1).float()
    return boundary


class HighF1CompoundLoss(nn.Module):
    """
    F1-Optimized Loss: 0.8 * Weighted CE + 1.6 * Soft Dice + 0.3 * Boundary BCE.
    """
    def __init__(self, weights=None, num_classes=5, ignore_index=255):
        super().__init__()
        weight_tensor = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
        self.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ignore_index)
        self.dice_loss = SoftDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, semantic_logits, boundary_logits, targets):
        ce = self.ce_loss(semantic_logits, targets)
        dice = self.dice_loss(semantic_logits, targets)
        
        gt_boundary = extract_sobel_boundaries(targets, stride=4)
        bce = self.bce_loss(boundary_logits, gt_boundary)
        
        # 1.6x Dice weight aggressively optimizes class-wise overlap and F1
        total_loss = 0.8 * ce + 1.6 * dice + 0.3 * bce
        return total_loss, ce.item(), dice.item(), bce.item()


class StreamingMetrics:
    def __init__(self, num_classes=5, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix.fill(0)

    def update(self, preds, targets):
        preds = preds.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        
        valid = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        v_preds = preds[valid]
        v_targets = targets[valid]
        
        counts = np.bincount(
            self.num_classes * v_targets + v_preds,
            minlength=self.num_classes ** 2
        )
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self):
        ious = []
        f1_scores = []
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            fn = np.sum(self.confusion_matrix[c, :]) - tp
            denom_iou = tp + fp + fn
            iou = tp / denom_iou if denom_iou > 0 else 0.0
            ious.append(iou)

            denom_f1 = 2 * tp + fp + fn
            f1 = (2 * tp) / denom_f1 if denom_f1 > 0 else 0.0
            f1_scores.append(f1)
            
        total_correct = np.diag(self.confusion_matrix).sum()
        total_valid = self.confusion_matrix.sum()
        acc = total_correct / total_valid if total_valid > 0 else 0.0
        miou = float(np.mean(ious))
        mean_f1 = float(np.mean(f1_scores))
        return miou, mean_f1, acc, ious, f1_scores


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss, total_ce, total_dice, total_bce = 0.0, 0.0, 0.0, 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            semantic_logits, boundary_logits = model(images)
            loss, ce_val, dice_val, bce_val = criterion(semantic_logits, boundary_logits, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_ce += ce_val
        total_dice += dice_val
        total_bce += bce_val
        
        if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == num_batches:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch:02d}] Batch [{batch_idx+1:03d}/{num_batches:03d}] -> Loss: {avg_loss:.4f} (CE: {ce_val:.4f}, Dice: {dice_val:.4f})", flush=True)
            
    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device, metrics):
    model.eval()
    metrics.reset()
    total_loss = 0.0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            semantic_logits = model(images)
            loss = F.cross_entropy(semantic_logits, masks, ignore_index=IGNORE_INDEX)
            
        preds = torch.argmax(semantic_logits, dim=1)
        metrics.update(preds, masks)
        total_loss += loss.item()
        
    miou, mean_f1, acc, per_class_iou, per_class_f1 = metrics.compute()
    val_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return val_loss, miou, mean_f1, acc, per_class_iou, per_class_f1


def main():
    parser = argparse.ArgumentParser(description="Retrain Strategic Terrain Segmenter for High F1")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--fold", type=int, default=1, help="Cross-validation fold")
    parser.add_argument("--crop_size", type=int, default=512, help="Patch size")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is required for high-efficiency training.")
        return

    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    print(f"Using GPU: {torch.cuda.get_device_name(device)} (VRAM safe-budget: ~5.6 GB)", flush=True)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"\nInitializing DataLoaders for Fold {args.fold} (crop size {args.crop_size}x{args.crop_size})...")
    train_loader, val_loader = get_segmentation_loaders(
        fold=args.fold,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=0
    )

    # Model
    model = TerrainSegmenter(num_classes=NUM_CLASSES).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TerrainSegmenter Parameters: {params:,}")

    # F1-Optimized Loss & Optimizer
    criterion = HighF1CompoundLoss(
        weights=CLASS_WEIGHTS,
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    metrics = StreamingMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    checkpoint_dir = root_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / f"terrain_segmenter_fold{args.fold}_best.pt"

    # Pre-load existing weights to accelerate convergence
    if best_ckpt_path.exists():
        try:
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print(f"Loaded existing checkpoint from {best_ckpt_path} to refine toward higher F1!", flush=True)
        except Exception as e:
            print(f"Starting fresh weights: {e}", flush=True)

    best_f1 = 0.0
    print("\n" + "=" * 70)
    print(f"   STARTING HIGH-F1 TERRAIN SEGMENTATION RETRAINING (FOLD {args.fold})")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        scheduler.step()
        
        val_loss, miou, mean_f1, acc, class_ious, class_f1s = validate(model, val_loader, criterion, device, metrics)
        elapsed = time.time() - t0
        
        print(f"\n--- Epoch {epoch:02d}/{args.epochs:02d} ({elapsed:.1f}s) ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {acc*100:.2f}% | Mean F1: {mean_f1*100:.2f}% | mIoU: {miou*100:.2f}%")
        print("Per-Class Scores:")
        for c, name in enumerate(CLASS_NAMES):
            print(f"  {name:12s} -> F1: {class_f1s[c]*100:5.2f}% | IoU: {class_ious[c]*100:5.2f}%")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_f1": best_f1,
                "miou": miou,
                "class_ious": class_ious,
                "class_f1s": class_f1s,
                "fold": args.fold
            }, best_ckpt_path)
            print(f"  >>> HIGHER F1 ACHIEVED: Mean F1 = {best_f1*100:.2f}% (mIoU = {miou*100:.2f}%) -> Saved!\n")

    print("=" * 70)
    print(f"High-F1 Training Complete! Best Mean F1-Score: {best_f1*100:.2f}%")
    print(f"Saved Checkpoint: {best_ckpt_path}")
    print("=" * 70)

    # Export to static ONNX
    onnx_path = checkpoint_dir / "terrain_segmenter_512.onnx"
    print(f"\nExporting High-F1 Terrain Segmenter to ONNX: {onnx_path}...")
    model.eval()
    dummy_input = torch.randn(1, 3, args.crop_size, args.crop_size, device=device)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input_tile"],
            output_names=["semantic_logits", "boundary_logits"],
            opset_version=18,
            do_constant_folding=True
        )
        print(f"Successfully exported ONNX ({onnx_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"ONNX export warning: {e}")


if __name__ == "__main__":
    main()
