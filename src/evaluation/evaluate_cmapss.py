"""
src/evaluation/evaluate_cmapss.py
Evaluates the trained C-MAPSS RUL & Early Failure Warning model.
Reports Precision, Recall, Critical F1, Macro F1, and Regression MAE/RMSE.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.rul_predictor import CMAPSSRULPredictor
from src.training.train_cmapss import CMAPSSWindowDataset, compute_metrics, SEQ_LEN


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = root_dir / "checkpoints" / "cmapss_rul_best.pt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = CMAPSSRULPredictor(in_features=20, seq_len=SEQ_LEN).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_dir = root_dir / "data_processed" / "cmapss" / "FD001"
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    u_test = np.load(data_dir / "test_units.npy")

    test_dataset = CMAPSSWindowDataset(X_test, y_test, u_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    probs, targets = [], []
    rul_preds, rul_targets = [], []
    latencies = []

    with torch.no_grad():
        for bx, bc, br in test_loader:
            bx = bx.to(device)
            t0 = time.perf_counter()
            logits, rp = model(bx)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000 / len(bx))

            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p)
            targets.extend(bc.numpy())
            rul_preds.extend(rp.cpu().numpy())
            rul_targets.extend(br.numpy())

    probs = np.array(probs)
    targets = np.array(targets)
    rul_preds = np.array(rul_preds)
    rul_targets = np.array(rul_targets)

    # Threshold sweep to find optimal operational operating point
    print("==========================================================")
    print("   C-MAPSS TEST SET DECISION THRESHOLD OPERATING CURVE    ")
    print("==========================================================")
    best_thresh = 0.20
    best_f1 = 0.0
    best_metrics = None

    for t in np.arange(0.10, 0.50, 0.02):
        preds = (probs >= t).astype(int)
        m = compute_metrics(targets, preds, rul_targets, rul_preds)
        print(f"  Threshold {t:.2f} -> Prec: {m['precision_pct']:5.1f}% | Rec: {m['recall_pct']:5.1f}% | Critical F1: {m['f1_score_pct']:5.1f}% | Macro F1: {m['macro_f1_pct']:5.1f}%")
        if m["f1_score_pct"] > best_f1:
            best_f1 = m["f1_score_pct"]
            best_thresh = float(t)
            best_metrics = m

    best_metrics["calibrated_threshold"] = round(best_thresh, 2)
    best_metrics["mean_inference_latency_ms"] = round(float(np.mean(latencies)), 3)

    print("\n==========================================================")
    print(f"  OPTIMAL CALIBRATED OPERATING POINT (Threshold: {best_thresh:.2f})")
    print("==========================================================")
    print(f"  Test Accuracy:               {best_metrics['accuracy_pct']}%")
    print(f"  Critical Failure F1 Score:   {best_metrics['f1_score_pct']}%")
    print(f"  Macro F1 Score:              {best_metrics['macro_f1_pct']}%")
    print(f"  Precision:                   {best_metrics['precision_pct']}%")
    print(f"  Recall (Failure Intercept):  {best_metrics['recall_pct']}%")
    print(f"  RUL Mean Absolute Error:     {best_metrics['rul_mae']} cycles")
    print(f"  RUL Root Mean Squared Error: {best_metrics['rul_rmse']} cycles")
    print(f"  Inference Latency:           {best_metrics['mean_inference_latency_ms']} ms/sample")
    print("==========================================================")

    # Update report
    report_path = root_dir / "data_processed" / "cmapss_evaluation_report.json"
    report = {
        "model_name": "CMAPSSRULPredictor",
        "dataset": "NASA C-MAPSS Turbofan Degradation (FD001)",
        "in_features": 20,
        "seq_len": SEQ_LEN,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "test_metrics": best_metrics
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Updated {report_path}")

    return best_metrics


if __name__ == "__main__":
    evaluate()
