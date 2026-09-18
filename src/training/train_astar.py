"""
src/training/train_astar.py
High-Efficiency GPU Training Engine for 3D Kinodynamic Neural A* Planner.
- Safeguarded for Laptop GPUs: caps VRAM allocation to 70% (~5.6 GB) to ensure OS display stability.
- Utilizes full CUDA Tensor Core throughput via AMP FP16 and TensorFloat-32.
- In-memory tensor generation for maximum GPU feeding speed (zero CPU bottleneck).
- Exports static ONNX model upon completion.
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models.kinematic_astar import KinematicAStarNet, NUM_PRIMITIVES


def generate_synthetic_kinodynamic_tensors(num_samples=3000, map_size=256):
    """
    Pre-generates all planning tensors in contiguous memory for high-speed GPU streaming.
    """
    print(f"Pre-generating {num_samples:,} 3D kinodynamic planning tensors in RAM...", flush=True)
    H, W = map_size, map_size
    
    # 1. Costmaps: (N, 3, H, W)
    costmaps = np.full((num_samples, 3, H, W), 0.20, dtype=np.float32)
    
    for i in range(num_samples):
        # Water hazard (Channel 0)
        wx, wy = np.random.randint(20, W - 20), np.random.randint(20, H - 20)
        wr = np.random.randint(20, 50)
        y_g, x_g = np.ogrid[:H, :W]
        costmaps[i, 0, (x_g - wx)**2 + (y_g - wy)**2 <= wr**2] = 1.00
        
        # Road corridor (Channel 0)
        rx = np.random.randint(40, W - 40)
        costmaps[i, 0, :, max(0, rx-6):min(W, rx+6)] = 0.05
        
        # Obstacles (Channel 1)
        for _ in range(np.random.randint(8, 20)):
            bx = np.random.randint(10, W - 30)
            by = np.random.randint(10, H - 30)
            bw, bh = np.random.randint(8, 24), np.random.randint(8, 24)
            costmaps[i, 1, by:by+bh, bx:bx+bw] = 0.90
            
        # Altitude gradient (Channel 2)
        costmaps[i, 2, :, :] = np.linspace(0.1, 0.9, H, dtype=np.float32)[:, None]

    # 2. States & Goals
    sx = np.random.uniform(10, W - 10, size=num_samples).astype(np.float32)
    sy = np.random.uniform(10, H - 10, size=num_samples).astype(np.float32)
    sz = np.random.uniform(5, 40, size=num_samples).astype(np.float32)
    vx = np.random.uniform(-4, 4, size=num_samples).astype(np.float32)
    vy = np.random.uniform(-4, 4, size=num_samples).astype(np.float32)
    vz = np.random.uniform(-1, 1, size=num_samples).astype(np.float32)
    states = np.stack([sx, sy, sz, vx, vy, vz], axis=1)

    gx = np.random.uniform(10, W - 10, size=num_samples).astype(np.float32)
    gy = np.random.uniform(10, H - 10, size=num_samples).astype(np.float32)
    gz = np.random.uniform(5, 40, size=num_samples).astype(np.float32)
    goals = np.stack([gx, gy, gz], axis=1)

    # 3. Targets
    target_heuristics = np.zeros((num_samples, 1, H, W), dtype=np.float32)
    target_actions = np.zeros(num_samples, dtype=np.int64)
    target_safeties = np.zeros((num_samples, 1), dtype=np.float32)

    for i in range(num_samples):
        # Heuristic field
        dist = np.sqrt((x_g - gx[i])**2 + (y_g - gy[i])**2 + (sz[i] - gz[i])**2) / 100.0
        risk = costmaps[i, 0] * 1.5 + costmaps[i, 1] * 3.0
        target_heuristics[i, 0] = dist + risk
        
        # Action primitive
        dx = gx[i] - sx[i]
        dy = gy[i] - sy[i]
        dz = gz[i] - sz[i]
        norm = max(1e-4, math.sqrt(dx*dx + dy*dy + dz*dz))
        dir_x = 1 if dx / norm > 0.3 else (-1 if dx / norm < -0.3 else 0)
        dir_y = 1 if dy / norm > 0.3 else (-1 if dy / norm < -0.3 else 0)
        dir_z = 1 if dz / norm > 0.3 else (-1 if dz / norm < -0.3 else 0)
        act = (dir_x + 1) * 9 + (dir_y + 1) * 3 + (dir_z + 1)
        target_actions[i] = min(26, max(0, int(act)))
        
        # Safety
        cx_idx = int(np.clip(sx[i], 0, W-1))
        cy_idx = int(np.clip(sy[i], 0, H-1))
        safe = 1.0 if costmaps[i, 1, cy_idx, cx_idx] < 0.5 and costmaps[i, 0, cy_idx, cx_idx] < 0.8 else 0.0
        target_safeties[i, 0] = safe

    print(f"Generated {num_samples:,} tensors ({costmaps.nbytes / (1024*1024):.1f} MB costmaps). Ready for GPU streaming.", flush=True)
    return (
        torch.from_numpy(costmaps),
        torch.from_numpy(states),
        torch.from_numpy(goals),
        torch.from_numpy(target_heuristics),
        torch.from_numpy(target_actions),
        torch.from_numpy(target_safeties)
    )


def train_kinematic_astar(
    epochs=10,
    batch_size=48,
    map_size=256,
    lr=1e-3,
    num_samples=3200,
    export_onnx=True
):
    print("=" * 70, flush=True)
    print("   TRAINING 3D KINEMATIC A* PLANNER ON NVIDIA GPU (SAFE HIGH-EFFICIENCY)", flush=True)
    print("=" * 70, flush=True)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", flush=True)
        return

    dev = torch.device("cuda:0")
    # Laptop stability safeguard: cap VRAM allocation to 70% (~5.6 GB) to preserve Windows DWM buffer
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    print(f"Device: {torch.cuda.get_device_name(dev)}", flush=True)
    print("VRAM Budget: Capped at 70% (~5.6 GB) for rock-solid system stability", flush=True)

    # Enable Tensor Core optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Pre-generate dataset
    t_data = generate_synthetic_kinodynamic_tensors(num_samples=num_samples, map_size=map_size)
    dataset = TensorDataset(*t_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    print(f"Dataset: {len(dataset):,} planning instances | Batch size: {batch_size} | Batches/epoch: {len(loader)}", flush=True)

    # Model
    model = KinematicAStarNet(base_channels=32, num_primitives=NUM_PRIMITIVES).to(dev)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"KinematicAStarNet Parameters: {params:,}", flush=True)

    criterion_heuristic = nn.SmoothL1Loss()
    criterion_policy = nn.CrossEntropyLoss()
    criterion_safety = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    ckpt_dir = root_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "kinematic_astar_best.pt"

    print("\nStarting GPU Training Loop with Automatic Mixed Precision (AMP)...", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_h, total_p, total_s = 0.0, 0.0, 0.0, 0.0

        for batch_idx, (costmaps, states, goals, target_h, target_p, target_s) in enumerate(loader):
            costmaps = costmaps.to(dev, non_blocking=True)
            states = states.to(dev, non_blocking=True)
            goals = goals.to(dev, non_blocking=True)
            target_h = target_h.to(dev, non_blocking=True)
            target_p = target_p.to(dev, non_blocking=True)
            target_s = target_s.to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred_h, pred_p, pred_s = model(costmaps, states, goals)
                loss_h = criterion_heuristic(pred_h, target_h)
                loss_p = criterion_policy(pred_p, target_p)
                loss_s = criterion_safety(pred_s, target_s)
                loss = loss_h + 0.5 * loss_p + 0.5 * loss_s

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_h += loss_h.item()
            total_p += loss_p.item()
            total_s += loss_s.item()

            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(loader):
                cur_vram = torch.cuda.memory_allocated(dev) / (1024**2)
                max_vram = torch.cuda.max_memory_allocated(dev) / (1024**2)
                avg_l = total_loss / (batch_idx + 1)
                print(f"Epoch [{epoch:02d}/{epochs:02d}] Batch [{batch_idx+1:03d}/{len(loader):03d}] -> Loss: {avg_l:.4f} (Heur: {total_h/(batch_idx+1):.4f}, Policy: {total_p/(batch_idx+1):.4f}, Safe: {total_s/(batch_idx+1):.4f}) | VRAM: {cur_vram:.0f} MB (Peak: {max_vram:.0f} MB)", flush=True)

        scheduler.step()
        elapsed = time.time() - t0
        avg_epoch_loss = total_loss / len(loader)
        print(f"\n--- Epoch {epoch:02d} Complete ({elapsed:.1f}s) | Avg Loss: {avg_epoch_loss:.4f} ---\n", flush=True)

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": avg_epoch_loss,
            "parameters": params
        }, best_ckpt)

    print("=" * 70, flush=True)
    print("Kinematic A* Planner Training Successfully Completed!")
    print(f"Saved Checkpoint: {best_ckpt}", flush=True)
    print("=" * 70, flush=True)

    if export_onnx:
        onnx_path = ckpt_dir / f"kinematic_astar_{map_size}.onnx"
        print(f"\nExporting Kinematic A* Planner to static ONNX: {onnx_path}...", flush=True)
        model.eval()
        dummy_c = torch.randn(1, 3, map_size, map_size, device=dev)
        dummy_s = torch.randn(1, 6, device=dev)
        dummy_g = torch.randn(1, 3, device=dev)

        try:
            torch.onnx.export(
                model,
                (dummy_c, dummy_s, dummy_g),
                str(onnx_path),
                input_names=["costmap", "state", "goal"],
                output_names=["heuristic_field", "primitive_policy", "safety_logits"],
                opset_version=17,
                do_constant_folding=True
            )
            print(f"Successfully exported ONNX: {onnx_path.stat().st_size / (1024*1024):.2f} MB", flush=True)
        except Exception as e:
            print(f"ONNX export warning: {e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Kinematic A* Planner on GPU")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")
    parser.add_argument("--map_size", type=int, default=256, help="Costmap resolution")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--samples", type=int, default=3200, help="Number of dataset samples")
    args = parser.parse_args()

    train_kinematic_astar(
        epochs=args.epochs,
        batch_size=args.batch_size,
        map_size=args.map_size,
        lr=args.lr,
        num_samples=args.samples,
        export_onnx=True
    )
