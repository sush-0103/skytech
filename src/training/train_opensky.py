"""
src/training/train_opensky.py
High-Efficiency Training Engine for OpenSky Airspace Traffic & Conflict Predictor.
- Safeguarded for Laptop GPUs: enforces 70% VRAM cap on RTX 5070 Laptop GPU.
- Utilizes full CUDA Tensor Core throughput with AMP FP16 and TensorFloat-32.
- Trains for 15 Epochs with Alpha-Balanced Focal Loss + Smooth L1 Trajectory Loss.
- Implements peak F1 threshold optimization on held-out validation aircraft.
- Exports best checkpoint (.pt) and static ONNX model upon completion.
"""

import os
import sys
import time
import math
import json
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

from src.models.traffic_predictor import AirspaceTrafficPredictor
from src.data.opensky_dataset import get_opensky_loaders


def binary_focal_loss(logits, targets, alpha=0.65, gamma=2.0):
    """
    Alpha-Balanced Focal Loss for addressing class imbalance in conflict classification.
    Forces model to focus on hard conflict boundaries and eliminates false negatives.
    """
    bce = F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets, reduction="none")
    p_t = torch.exp(-bce)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = alpha_t * ((1.0 - p_t) ** gamma) * bce
    return focal.mean()


def evaluate_val_set(model, val_loader, device="cuda"):
    """
    Evaluates model across validation set and scans decision thresholds to find peak F1.
    """
    model.eval()
    all_probs = []
    all_targets = []
    all_traj_losses = []

    with torch.no_grad():
        for hist_feats, conflict, fut_offsets in val_loader:
            hist_feats = hist_feats.to(device)
            conflict = conflict.to(device)
            fut_offsets = fut_offsets.to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                c_logits, traj_preds = model(hist_feats)
                probs = torch.sigmoid(c_logits.squeeze(-1))
                traj_loss = F.smooth_l1_loss(traj_preds, fut_offsets)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(conflict.cpu().numpy().tolist())
            all_traj_losses.append(traj_loss.item())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    mean_traj_loss = float(np.mean(all_traj_losses))

    # Scan decision thresholds to locate maximum F1 score
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_acc = 0.0
    best_thresh = 0.50
    best_tp, best_fp, best_fn, best_tn = 0, 0, 0, 0

    for thresh in np.linspace(0.20, 0.80, 61):
        preds = (all_probs >= thresh).astype(np.float32)
        tp = int(np.sum((preds == 1) & (all_targets == 1)))
        fp = int(np.sum((preds == 1) & (all_targets == 0)))
        fn = int(np.sum((preds == 0) & (all_targets == 1)))
        tn = int(np.sum((preds == 0) & (all_targets == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(all_targets) if len(all_targets) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_acc = acc
            best_thresh = float(thresh)
            best_tp, best_fp, best_fn, best_tn = tp, fp, fn, tn

    return {
        "f1": best_f1,
        "precision": best_prec,
        "recall": best_rec,
        "accuracy": best_acc,
        "optimal_threshold": best_thresh,
        "tp": best_tp,
        "fp": best_fp,
        "fn": best_fn,
        "tn": best_tn,
        "traj_loss": mean_traj_loss
    }


def train_opensky_predictor(epochs=15, batch_size=32, lr=1e-3, device="cuda"):
    print("=================================================================")
    print("   OPENSKY AIR TRAFFIC & CONFLICT PREDICTOR TRAINING ENGINE      ")
    print("=================================================================")
    
    # Enforce Laptop GPU VRAM cap (70% ~ 5.6 GB)
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.70, 0)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"[Hardware Safeguard] VRAM allocation capped at 70% on {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"[Warning] Failed to set memory fraction: {e}")
    else:
        device = "cpu"
        print("[Device] Running on CPU")

    # Dataloaders
    train_loader, val_loader = get_opensky_loaders(batch_size=batch_size)
    print(f"Data ready: {len(train_loader)} train batches, {len(val_loader)} validation batches.")

    # Model
    model = AirspaceTrafficPredictor(in_features=8, hidden_dim=128, fut_steps=3).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: AirspaceTrafficPredictor ({param_count:,} parameters).")

    # Optimizer & LR Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    out_dir = root_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "airspace_predictor_best.pt"

    best_val_f1 = 0.0
    best_metrics = {}

    print(f"\nStarting {epochs}-Epoch Training Loop on {device.upper()}...\n")
    print(f"{'Epoch':<7} | {'Train Loss':<11} | {'Val F1':<9} | {'Val Prec':<9} | {'Val Rec':<9} | {'Val Acc':<9} | {'LR':<9} | {'Time':<6}")
    print("-" * 75)

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []

        for hist_feats, conflict, fut_offsets in train_loader:
            hist_feats = hist_feats.to(device, non_blocking=True)
            conflict = conflict.to(device, non_blocking=True)
            fut_offsets = fut_offsets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                c_logits, traj_preds = model(hist_feats)
                focal = binary_focal_loss(c_logits, conflict, alpha=0.65, gamma=2.0)
                traj_loss = F.smooth_l1_loss(traj_preds, fut_offsets)
                total_loss = focal + 0.35 * traj_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(total_loss.item())

        scheduler.step()
        train_loss_mean = float(np.mean(train_losses))
        dt = time.time() - t0

        # Evaluate on validation set
        val_res = evaluate_val_set(model, val_loader, device=device)
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"{epoch:<7} | "
            f"{train_loss_mean:<11.4f} | "
            f"{val_res['f1']*100:<8.2f}% | "
            f"{val_res['precision']*100:<8.2f}% | "
            f"{val_res['recall']*100:<8.2f}% | "
            f"{val_res['accuracy']*100:<8.2f}% | "
            f"{cur_lr:<9.2e} | "
            f"{dt:<5.1f}s"
        )

        if val_res["f1"] > best_val_f1:
            best_val_f1 = val_res["f1"]
            best_metrics = val_res
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "f1": val_res["f1"],
                "precision": val_res["precision"],
                "recall": val_res["recall"],
                "accuracy": val_res["accuracy"],
                "optimal_threshold": val_res["optimal_threshold"],
                "loss": train_loss_mean
            }, str(best_ckpt_path))

    total_duration = time.time() - start_total_time
    print("-" * 75)
    print(f"\nTraining completed in {total_duration:.1f}s.")
    print(f"\n=======================================================")
    print(f"             FINAL MODEL VALIDATION RESULTS            ")
    print(f"=======================================================")
    print(f"  Best Validation F1-Score: {best_metrics.get('f1', 0)*100:.2f}%")
    print(f"  Precision:               {best_metrics.get('precision', 0)*100:.2f}%")
    print(f"  Recall:                  {best_metrics.get('recall', 0)*100:.2f}%")
    print(f"  Overall Accuracy:        {best_metrics.get('accuracy', 0)*100:.2f}%")
    print(f"  Optimal Decision Thresh: {best_metrics.get('optimal_threshold', 0.5):.2f}")
    print(f"  True Positives (TP):     {best_metrics.get('tp', 0)}")
    print(f"  False Positives (FP):    {best_metrics.get('fp', 0)}")
    print(f"  False Negatives (FN):    {best_metrics.get('fn', 0)}")
    print(f"  True Negatives (TN):     {best_metrics.get('tn', 0)}")
    print(f"  Saved Checkpoint:        {best_ckpt_path}")

    # Export Static ONNX Engine
    onnx_path = out_dir / "airspace_predictor.onnx"
    try:
        model.eval().to("cpu")
        dummy_input = torch.randn(1, 4, 8)
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=["history_states"],
            output_names=["conflict_logits", "future_trajectory"],
            dynamic_axes={"history_states": {0: "batch_size"}},
            opset_version=17
        )
        print(f"  Exported ONNX Model:     {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"  Warning on ONNX export: {e}")

    # Save evaluation manifest
    manifest_path = out_dir / "opensky_evaluation.json"
    manifest = {
        "model": "AirspaceTrafficPredictor",
        "dataset": "OpenSky ADS-B Telemetry",
        "epochs": epochs,
        "best_f1_pct": round(best_metrics.get("f1", 0) * 100, 2),
        "precision_pct": round(best_metrics.get("precision", 0) * 100, 2),
        "recall_pct": round(best_metrics.get("recall", 0) * 100, 2),
        "accuracy_pct": round(best_metrics.get("accuracy", 0) * 100, 2),
        "optimal_threshold": best_metrics.get("optimal_threshold", 0.5),
        "confusion_matrix": {
            "tp": best_metrics.get("tp", 0),
            "fp": best_metrics.get("fp", 0),
            "fn": best_metrics.get("fn", 0),
            "tn": best_metrics.get("tn", 0)
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved Evaluation JSON:   {manifest_path}\n")


if __name__ == "__main__":
    train_opensky_predictor(epochs=15, batch_size=32, lr=1e-3, device="cuda")
