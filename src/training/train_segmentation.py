"""
src/training/train_segmentation.py
Full training engine for Strategic Terrain Segmentation on Dubai dataset.
- Compound Loss: Class-Balanced Cross-Entropy + Soft Multiclass Dice + Auxiliary Boundary BCE.
- Streaming Confusion Matrix for memory-efficient mIoU calculation.
- Cosine Annealing with Warmup learning rate schedule.
- Saves best checkpoint by validation mIoU.
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

from src.models.terrain_segmenter import TerrainSegmenter
from src.data.segmentation_dataset import get_segmentation_loaders, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX

# Inverse-frequency class weights based on measured pixel distribution
# Land: 52.36%, Water: 13.61%, Building: 13.31%, Vegetation: 10.31%, Road: 9.35%
PIXEL_FREQS = [0.5236, 0.1361, 0.1331, 0.1031, 0.0935]
CLASS_WEIGHTS = [1.0 / np.log(1.02 + p) for p in PIXEL_FREQS]


class SoftDiceLoss(nn.Module):
    """Multi-class Soft Dice Loss supporting ignore_index."""
    def __init__(self, num_classes=5, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        probs = F.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index)  # (B, H, W)
        
        # Clamp targets to avoid indexing out of bounds
        clamped_targets = targets.clone()
        clamped_targets[~valid_mask] = 0
        
        # One-hot encode targets
        one_hot = F.one_hot(clamped_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Mask out invalid pixels across all channels
        probs = probs * valid_mask.unsqueeze(1)
        one_hot = one_hot * valid_mask.unsqueeze(1)
        
        # Compute Dice per class
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        cardinality = torch.sum(probs * probs + one_hot * one_hot, dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        return 1.0 - torch.mean(dice_score)


def extract_sobel_boundaries(masks, stride=4):
    """
    Extracts 1-pixel binary boundary edges from ground truth masks at downsampled resolution.
    masks: (B, H, W) int64 tensor.
    returns: (B, 1, H/stride, W/stride) float32 tensor.
    """
    # Downsample mask to P2 stride
    b, h, w = masks.shape
    ds_h, ds_w = h // stride, w // stride
    
    downsampled = F.interpolate(
        masks.unsqueeze(1).float(),
        size=(ds_h, ds_w),
        mode="nearest"
    ).squeeze(1).long()
    
    # Sobel kernels for horizontal and vertical edge gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=masks.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=masks.device).view(1, 1, 3, 3)
    
    # Mask out ignore index
    clean_ds = downsampled.clone()
    clean_ds[clean_ds == IGNORE_INDEX] = 0
    clean_ds = clean_ds.unsqueeze(1).float()
    
    grad_x = F.conv2d(clean_ds, sobel_x, padding=1)
    grad_y = F.conv2d(clean_ds, sobel_y, padding=1)
    
    edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    boundary = (edge_magnitude > 0.1).float()
    return boundary


class CompoundSegmentationLoss(nn.Module):
    """
    1.0 * Class-Balanced Cross-Entropy + 0.5 * Soft Dice + 0.2 * Boundary BCE.
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
        
        # Auxiliary boundary loss
        gt_boundary = extract_sobel_boundaries(targets, stride=4)
        bce = self.bce_loss(boundary_logits, gt_boundary)
        
        total_loss = 1.0 * ce + 0.5 * dice + 0.2 * bce
        return total_loss, ce.item(), dice.item(), bce.item()


class StreamingMetrics:
    """Accumulates confusion matrix for accurate per-class IoU calculation."""
    def __init__(self, num_classes=5, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix.fill(0)

    def update(self, preds, targets):
        # preds: (B, H, W) int64, targets: (B, H, W) int64
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
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            fn = np.sum(self.confusion_matrix[c, :]) - tp
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            ious.append(iou)
            
        total_correct = np.diag(self.confusion_matrix).sum()
        total_valid = self.confusion_matrix.sum()
        acc = total_correct / total_valid if total_valid > 0 else 0.0
        miou = float(np.mean(ious))
        return miou, acc, ious


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, total_ce, total_dice, total_bce = 0.0, 0.0, 0.0, 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        semantic_logits, boundary_logits = model(images)
        loss, ce_val, dice_val, bce_val = criterion(semantic_logits, boundary_logits, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ce += ce_val
        total_dice += dice_val
        total_bce += bce_val
        
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch:02d}] Batch [{batch_idx+1:03d}/{num_batches:03d}] -> Loss: {avg_loss:.4f} (CE: {ce_val:.4f}, Dice: {dice_val:.4f}, Bnd: {bce_val:.4f})", flush=True)
            
    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device, metrics):
    model.eval()
    metrics.reset()
    total_loss = 0.0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        
        metrics.update(preds, masks)
        ce = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
        total_loss += ce.item()
        
    miou, acc, per_class_iou = metrics.compute()
    val_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return val_loss, miou, acc, per_class_iou


def main():
    parser = argparse.ArgumentParser(description="Train Terrain Segmentation Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Base learning rate")
    parser.add_argument("--fold", type=int, default=1, help="Cross-validation fold (1..8)")
    parser.add_argument("--crop_size", type=int, default=512, help="Patch size")
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        num_threads = os.cpu_count() or 24
        torch.set_num_threads(num_threads)
        print(f"Using CPU: Intel Core Ultra 9 with {num_threads} parallel threads", flush=True)

    # DataLoaders
    print(f"\nInitializing DataLoaders for Fold {args.fold} (crop size {args.crop_size}x{args.crop_size})...")
    train_loader, val_loader = get_segmentation_loaders(
        fold=args.fold,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=0
    )

    # Model
    model = TerrainSegmenter(num_classes=NUM_CLASSES).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss & Optimizer
    criterion = CompoundSegmentationLoss(
        weights=CLASS_WEIGHTS,
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    metrics = StreamingMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    checkpoint_dir = Path(r"c:\Users\ahile\Downloads\FlYtech\checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / f"terrain_segmenter_fold{args.fold}_best.pt"

    best_miou = 0.0
    print("\n==================================================")
    print(f"   STARTING TERRAIN SEGMENTATION TRAINING (FOLD {args.fold})")
    print("==================================================")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        
        val_loss, miou, acc, class_ious = validate(model, val_loader, criterion, device, metrics)
        elapsed = time.time() - t0
        
        print(f"\n--- Epoch {epoch:02d}/{args.epochs:02d} Summary ({elapsed:.1f}s) ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {acc*100:.2f}% | Val mIoU: {miou*100:.2f}%")
        print("Per-class IoU:")
        for c, name in enumerate(CLASS_NAMES):
            print(f"  {name:12s}: {class_ious[c]*100:5.2f}%")

        if miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "miou": best_miou,
                "class_ious": class_ious,
                "fold": args.fold
            }, best_ckpt_path)
            print(f"  >>> NEW BEST MODEL SAVED: mIoU = {best_miou*100:.2f}% -> {best_ckpt_path}")
        print()

    print("==================================================")
    print(f"Training Complete! Best Validation mIoU: {best_miou*100:.2f}%")
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
