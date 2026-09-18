"""
src/training/train_cmapss.py
Training pipeline for NASA C-MAPSS Turbofan Component Degradation & Predictive Maintenance.
- Dual-head model: Impending Failure Classification (F1 Score) + Continuous RUL Regression.
- Engine-level grouped cross-validation (80 train engines, 20 val engines, 100 test engines).
- Sliding window length: 30 cycles.
- Trained natively on NVIDIA GeForce RTX 5070 Laptop GPU (sm_120).
- Exports best model checkpoint and static ONNX graph.
"""

import os
import sys
import json
import time
import math
from pathlib import Path
import numpy as np

# Force UTF-8 on Windows terminal to prevent UnicodeEncodeError on emoji logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models.rul_predictor import CMAPSSRULPredictor

SEQ_LEN = 30
RUL_CRITICAL_THRESHOLD = 30.0  # Impending failure horizon (cycles)
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3


class CMAPSSWindowDataset(Dataset):
    """Sliding window dataset extracted per-engine with zero leakage across engines."""
    def __init__(self, X_features: np.ndarray, y_rul: np.ndarray, unit_ids: np.ndarray, selected_units=None):
        self.windows = []
        self.rul_targets = []
        self.class_targets = []

        all_units = np.unique(unit_ids)
        if selected_units is not None:
            target_units = [u for u in all_units if u in selected_units]
        else:
            target_units = list(all_units)

        for u in target_units:
            mask = unit_ids == u
            eng_x = X_features[mask]
            eng_y = y_rul[mask]

            n_cycles = len(eng_x)
            if n_cycles < SEQ_LEN:
                continue

            for t in range(SEQ_LEN - 1, n_cycles):
                w = eng_x[t - SEQ_LEN + 1 : t + 1]  # (SEQ_LEN, C)
                r = eng_y[t]
                cls_label = 1 if r <= RUL_CRITICAL_THRESHOLD else 0

                self.windows.append(w)
                self.rul_targets.append(r)
                self.class_targets.append(cls_label)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.rul_targets = np.array(self.rul_targets, dtype=np.float32)
        self.class_targets = np.array(self.class_targets, dtype=np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(self.class_targets[idx], dtype=torch.long),
            torch.tensor(self.rul_targets[idx], dtype=torch.float32)
        )


def compute_metrics(y_true, y_pred, y_rul_true, y_rul_pred):
    """Compute precision, recall, F1, accuracy, and regression MAE/RMSE."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-8, (precision + recall))

    # Macro F1
    prec_neg = tn / max(1, (tn + fn))
    rec_neg = tn / max(1, (tn + fp))
    f1_neg = 2 * prec_neg * rec_neg / max(1e-8, (prec_neg + rec_neg))
    macro_f1 = (f1 + f1_neg) / 2.0

    mae = np.mean(np.abs(y_rul_pred - y_rul_true))
    rmse = np.sqrt(np.mean((y_rul_pred - y_rul_true) ** 2))

    return {
        "accuracy_pct": round(accuracy * 100.0, 2),
        "precision_pct": round(precision * 100.0, 2),
        "recall_pct": round(recall * 100.0, 2),
        "f1_score_pct": round(f1 * 100.0, 2),
        "macro_f1_pct": round(macro_f1 * 100.0, 2),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "rul_mae": round(float(mae), 2),
        "rul_rmse": round(float(rmse), 2)
    }


def train_model():
    print("==================================================================")
    print("  NASA C-MAPSS PREDICTIVE MAINTENANCE: TRAINING DUAL-HEAD RUL NN  ")
    print("==================================================================")

    data_dir = root_dir / "data_processed" / "cmapss" / "FD001"
    checkpoints_dir = root_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load preprocessed arrays
    X_train_raw = np.load(data_dir / "X_train.npy")
    y_train_raw = np.load(data_dir / "y_train.npy")
    train_units_raw = np.load(data_dir / "train_units.npy")

    X_test_raw = np.load(data_dir / "X_test.npy")
    y_test_raw = np.load(data_dir / "y_test.npy")
    test_units_raw = np.load(data_dir / "test_units.npy")

    # Engine-level grouped split (80 train engines, 20 val engines)
    unique_train_engines = np.unique(train_units_raw)
    np.random.seed(42)
    shuffled_engines = np.random.permutation(unique_train_engines)
    train_eng_ids = set(shuffled_engines[:80])
    val_eng_ids = set(shuffled_engines[80:])

    print(f"[Dataset] Train Engines: {len(train_eng_ids)} | Val Engines: {len(val_eng_ids)} | Test Engines: {len(np.unique(test_units_raw))}")

    # Build datasets
    train_dataset = CMAPSSWindowDataset(X_train_raw, y_train_raw, train_units_raw, selected_units=train_eng_ids)
    val_dataset = CMAPSSWindowDataset(X_train_raw, y_train_raw, train_units_raw, selected_units=val_eng_ids)
    test_dataset = CMAPSSWindowDataset(X_test_raw, y_test_raw, test_units_raw)

    print(f"[Windows] Train Windows: {len(train_dataset):,} | Val Windows: {len(val_dataset):,} | Test Windows: {len(test_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Training device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Initialize model
    in_features = X_train_raw.shape[1]
    model = CMAPSSRULPredictor(in_features=in_features, seq_len=SEQ_LEN).to(device)

    # Class weighting for early failure detection
    cls_weights = torch.tensor([1.0, 3.5], device=device)
    cls_loss_fn = nn.CrossEntropyLoss(weight=cls_weights)
    reg_loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_f1 = 0.0
    best_epoch = 0
    checkpoint_path = checkpoints_dir / "cmapss_rul_best.pt"

    print("\nStarting training loop (15 Epochs)...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_reg_loss = 0.0

        for batch_x, batch_cls, batch_rul in train_loader:
            batch_x = batch_x.to(device)
            batch_cls = batch_cls.to(device)
            batch_rul = batch_rul.to(device)

            optimizer.zero_grad()
            logits, rul_pred = model(batch_x)

            loss_c = cls_loss_fn(logits, batch_cls)
            loss_r = reg_loss_fn(rul_pred, batch_rul)
            loss = loss_c + 0.05 * loss_r

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * len(batch_x)
            train_cls_loss += loss_c.item() * len(batch_x)
            train_reg_loss += loss_r.item() * len(batch_x)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        val_rul_preds, val_rul_targets = [], []

        with torch.no_grad():
            for batch_x, batch_cls, batch_rul in val_loader:
                batch_x = batch_x.to(device)
                logits, rul_pred = model(batch_x)

                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_cls.numpy())

                val_rul_preds.extend(rul_pred.cpu().numpy())
                val_rul_targets.extend(batch_rul.numpy())

        val_metrics = compute_metrics(
            np.array(val_targets), np.array(val_preds),
            np.array(val_rul_targets), np.array(val_rul_preds)
        )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy_pct']}% | "
            f"Val F1: {val_metrics['f1_score_pct']}% (Macro: {val_metrics['macro_f1_pct']}%) | "
            f"Val Recall: {val_metrics['recall_pct']}% | "
            f"RUL MAE: {val_metrics['rul_mae']} cyc"
        )

        if val_metrics["f1_score_pct"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score_pct"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "in_features": in_features,
                "seq_len": SEQ_LEN
            }, checkpoint_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best Epoch: {best_epoch} with Val F1: {best_val_f1}%")

    # 3. Final Independent Test Evaluation
    print("\n==================================================================")
    print("      INDEPENDENT TEST EVALUATION (100 HELD-OUT TEST ENGINES)     ")
    print("==================================================================")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Determine optimal threshold on validation set
    val_probs, val_true_cls = [], []
    with torch.no_grad():
        for batch_x, batch_cls, _ in val_loader:
            batch_x = batch_x.to(device)
            logits, _ = model(batch_x)
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            val_probs.extend(p)
            val_true_cls.extend(batch_cls.numpy())
    
    val_probs = np.array(val_probs)
    val_true_cls = np.array(val_true_cls)
    
    best_thresh = 0.5
    best_f1_opt = 0.0
    for t_cand in np.arange(0.20, 0.70, 0.02):
        pred_cand = (val_probs >= t_cand).astype(int)
        tp = np.sum((pred_cand == 1) & (val_true_cls == 1))
        fp = np.sum((pred_cand == 1) & (val_true_cls == 0))
        fn = np.sum((pred_cand == 0) & (val_true_cls == 1))
        f1_c = 2 * tp / max(1e-8, (2 * tp + fp + fn))
        if f1_c > best_f1_opt:
            best_f1_opt = f1_c
            best_thresh = float(t_cand)

    print(f"[Calibrator] Optimal Failure Detection Threshold: {best_thresh:.2f} (Val F1: {best_f1_opt*100:.2f}%)")

    test_preds, test_targets = [], []
    test_rul_preds, test_rul_targets = [], []
    test_latencies = []

    with torch.no_grad():
        for batch_x, batch_cls, batch_rul in test_loader:
            batch_x = batch_x.to(device)
            t0 = time.perf_counter()
            logits, rul_pred = model(batch_x)
            t1 = time.perf_counter()
            test_latencies.append((t1 - t0) * 1000 / len(batch_x))

            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= best_thresh).astype(int)
            test_preds.extend(preds)
            test_targets.extend(batch_cls.numpy())

            test_rul_preds.extend(rul_pred.cpu().numpy())
            test_rul_targets.extend(batch_rul.numpy())

    test_metrics = compute_metrics(
        np.array(test_targets), np.array(test_preds),
        np.array(test_rul_targets), np.array(test_rul_preds)
    )
    test_metrics["calibrated_threshold"] = round(best_thresh, 2)
    mean_latency_ms = float(np.mean(test_latencies))
    test_metrics["mean_inference_latency_ms"] = round(mean_latency_ms, 3)

    print(f"  Test Accuracy:               {test_metrics['accuracy_pct']}%")
    print(f"  Critical Failure F1 Score:   {test_metrics['f1_score_pct']}%")
    print(f"  Macro F1 Score:              {test_metrics['macro_f1_pct']}%")
    print(f"  Precision:                   {test_metrics['precision_pct']}%")
    print(f"  Recall (Failure Intercept):  {test_metrics['recall_pct']}%")
    print(f"  RUL Mean Absolute Error:     {test_metrics['rul_mae']} cycles")
    print(f"  RUL Root Mean Squared Error: {test_metrics['rul_rmse']} cycles")
    print(f"  Inference Latency:           {test_metrics['mean_inference_latency_ms']} ms/sample")
    print("==================================================================")

    # 4. Export to static ONNX graph
    onnx_path = checkpoints_dir / "cmapss_rul.onnx"
    dummy_input = torch.randn(1, SEQ_LEN, in_features, device=device)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["sensor_sequence"],
            output_names=["failure_logits", "rul_cycles"],
            dynamic_axes={"sensor_sequence": {0: "batch_size"}, "failure_logits": {0: "batch_size"}, "rul_cycles": {0: "batch_size"}},
            opset_version=17,
            do_constant_folding=True
        )
        print(f"Exported static ONNX graph to: {onnx_path}")
    except Exception as e:
        print(f"[ONNX Notice] Exporting with fallback script...")
        torch.onnx.export(
            model.cpu(),
            dummy_input.cpu(),
            str(onnx_path),
            export_params=True,
            opset_version=17
        )
        print(f"Fallback ONNX export successful: {onnx_path}")

    # 5. Save comprehensive evaluation report
    report = {
        "model_name": "CMAPSSRULPredictor",
        "dataset": "NASA C-MAPSS Turbofan Degradation (FD001)",
        "in_features": in_features,
        "seq_len": SEQ_LEN,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "train_time_seconds": round(elapsed, 1),
        "test_metrics": test_metrics
    }
    report_path = root_dir / "data_processed" / "cmapss_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to: {report_path}")

    return test_metrics


if __name__ == "__main__":
    train_model()
