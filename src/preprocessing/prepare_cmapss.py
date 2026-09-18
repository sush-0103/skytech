"""
prepare_cmapss.py
Preprocessing pipeline for NASA C-MAPSS Turbofan Engine Degradation dataset.
- Computes Remaining Useful Life (RUL) with piecewise-linear clipping (cap at 125 cycles).
- Calculates per-sensor normalization parameters strictly on training engines.
- Drops zero-variance sensor columns.
- Exports clean NumPy arrays (X_train, y_train, X_test, y_test) and metadata.
"""

import os
import json
from pathlib import Path
import numpy as np

RUL_CAP = 125

COLUMN_NAMES = [
    "unit_id", "cycle", "setting_1", "setting_2", "setting_3",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"
]


def load_txt_data(filepath: Path):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                data.append([float(x) for x in parts])
    return np.array(data, dtype=np.float32)


def preprocess_cmapss(raw_dir: Path, output_dir: Path):
    print("==================================================")
    print("        NASA C-MAPSS PREDICTIVE MAINTENANCE       ")
    print("==================================================")
    
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    manifest = {}
    
    for sub in subsets:
        print(f"\nProcessing subset {sub}...")
        sub_out = output_dir / sub
        sub_out.mkdir(parents=True, exist_ok=True)
        
        train_file = raw_dir / f"train_{sub}.txt"
        test_file = raw_dir / f"test_{sub}.txt"
        rul_file = raw_dir / f"RUL_{sub}.txt"
        
        train_raw = load_txt_data(train_file)
        test_raw = load_txt_data(test_file)
        rul_ground_truth = load_txt_data(rul_file).flatten()
        
        # 1. Compute RUL for training engines
        train_units = np.unique(train_raw[:, 0])
        train_rul = np.zeros(len(train_raw), dtype=np.float32)
        
        for u in train_units:
            mask = train_raw[:, 0] == u
            max_cycle = np.max(train_raw[mask, 1])
            train_rul[mask] = np.minimum(max_cycle - train_raw[mask, 1], RUL_CAP)
            
        # 2. Compute RUL for test engines
        test_units = np.unique(test_raw[:, 0])
        test_rul = np.zeros(len(test_raw), dtype=np.float32)
        
        for idx, u in enumerate(test_units):
            mask = test_raw[:, 0] == u
            last_cycle = np.max(test_raw[mask, 1])
            true_final_rul = rul_ground_truth[idx]
            test_rul[mask] = np.minimum((last_cycle - test_raw[mask, 1]) + true_final_rul, RUL_CAP)
            
        # 3. Feature selection: drop constant channels across train set
        sensor_data_train = train_raw[:, 2:]  # settings + 21 sensors (24 channels)
        stds = np.std(sensor_data_train, axis=0)
        active_indices = np.where(stds > 1e-4)[0]
        
        train_features = sensor_data_train[:, active_indices]
        test_features = test_raw[:, 2:][:, active_indices]
        
        # 4. Min-max normalization learned strictly on train set
        f_min = np.min(train_features, axis=0)
        f_max = np.max(train_features, axis=0)
        f_denom = np.where(f_max - f_min == 0, 1.0, f_max - f_min)
        
        train_norm = (train_features - f_min) / f_denom
        test_norm = (test_features - f_min) / f_denom
        
        # Save arrays
        np.save(sub_out / "X_train.npy", train_norm)
        np.save(sub_out / "y_train.npy", train_rul)
        np.save(sub_out / "X_test.npy", test_norm)
        np.save(sub_out / "y_test.npy", test_rul)
        np.save(sub_out / "train_units.npy", train_raw[:, 0].astype(np.int32))
        np.save(sub_out / "test_units.npy", test_raw[:, 0].astype(np.int32))
        
        meta = {
            "subset": sub,
            "train_engines": int(len(train_units)),
            "test_engines": int(len(test_units)),
            "train_cycles": int(len(train_raw)),
            "test_cycles": int(len(test_raw)),
            "active_feature_count": int(len(active_indices)),
            "dropped_constant_features": int(24 - len(active_indices)),
            "rul_cap": RUL_CAP,
        }
        with open(sub_out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        manifest[sub] = meta
        print(f"  Train: {meta['train_engines']} engines, {meta['train_cycles']} cycles")
        print(f"  Test:  {meta['test_engines']} engines, {meta['test_cycles']} cycles")
        print(f"  Active features: {meta['active_feature_count']} (dropped {meta['dropped_constant_features']} constant channels)")
        
    manifest_path = output_dir / "cmapss_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n==================================================")
    print(f"C-MAPSS preprocessing complete!")
    print(f"Saved manifest to {manifest_path}")
    print("==================================================\n")
    return manifest


if __name__ == "__main__":
    raw_cmapss = Path(r"c:\Users\ahile\Downloads\FlYtech\Datasets\predictive maintenance-20260918T051215Z-1-001\predictive maintenance\01_NASA_CMAPSS_predictive_maintenance\01_NASA_CMAPSS_predictive_maintenance\CMAPSSData")
    out_cmapss = Path(r"c:\Users\ahile\Downloads\FlYtech\data_processed\cmapss")
    preprocess_cmapss(raw_cmapss, out_cmapss)
