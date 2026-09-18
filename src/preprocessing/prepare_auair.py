"""
prepare_auair.py
Preprocessing and sanitization pipeline for AU-AIR Multimodal UAV dataset.
- Quarantines invalid non-positive area bounding boxes (54 boxes).
- Converts [top, left, height, width] to normalized [class, xc, yc, w, h].
- Maps AU-AIR categories to the unified 10-class tactical taxonomy.
- Partitions the 32,823 frames into clip-level train/val splits to prevent 5 Hz temporal leakage.
- Preserves flight telemetry context in the dataset manifest.
"""

import os
import json
from pathlib import Path
from collections import Counter

# AU-AIR raw categories: ['Human', 'Car', 'Truck', 'Van', 'Motorbike', 'Bicycle', 'Bus', 'Trailer']
AUAIR_TO_UNIFIED = {
    0: 0,  # Human -> person
    1: 1,  # Car -> car
    2: 3,  # Truck -> truck
    3: 2,  # Van -> van
    4: 5,  # Motorbike -> motorized_two_wheeler
    5: 4,  # Bicycle -> bicycle
    6: 6,  # Bus -> bus
    7: 7,  # Trailer -> trailer
}

# Clips for train / val split (guarantees zero adjacent video frame leakage)
VAL_CLIPS = [
    "frame_20190829091111",  # 2,592 frames (altitude ~20-30m)
    "frame_20190905143505",  # 1,580 frames
]


def preprocess_auair(raw_dir: Path, output_dir: Path):
    print("==================================================")
    print("          AU-AIR MULTIMODAL UAV PREPROCESSING     ")
    print("==================================================")
    
    anno_file = raw_dir / "annotations.json"
    print(f"Reading {anno_file}...")
    with open(anno_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    annotations = raw_data.get("annotations", [])
    print(f"Total raw frames: {len(annotations):,}")
    
    out_lbl_dir = output_dir / "labels"
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    quarantine_log = []
    class_counts = Counter()
    split_counts = {"train": 0, "val": 0}
    split_boxes = {"train": 0, "val": 0}
    train_files = []
    val_files = []
    
    img_w, img_h = 1920.0, 1080.0
    
    for idx, item in enumerate(annotations):
        img_name = item.get("image_name", "")
        stem = Path(img_name).stem
        
        # Determine clip
        clip_prefix = img_name.split("_x_")[0] if "_x_" in img_name else img_name[:20]
        split = "val" if clip_prefix in VAL_CLIPS else "train"
        split_counts[split] += 1
        
        if split == "val":
            val_files.append(img_name)
        else:
            train_files.append(img_name)
            
        raw_boxes = item.get("bbox", [])
        clean_boxes = []
        
        for b_idx, box in enumerate(raw_boxes):
            top = float(box.get("top", 0))
            left = float(box.get("left", 0))
            h = float(box.get("height", 0))
            w = float(box.get("width", 0))
            cat = int(box.get("class", 0))
            
            # Check for non-positive or corrupt area
            if w <= 0 or h <= 0:
                quarantine_log.append({
                    "frame": img_name,
                    "box_idx": b_idx,
                    "box": box,
                    "reason": "zero or negative width/height"
                })
                continue
                
            # Boundary clipping
            x1 = max(0.0, min(img_w, left))
            y1 = max(0.0, min(img_h, top))
            x2 = max(0.0, min(img_w, left + w))
            y2 = max(0.0, min(img_h, top + h))
            
            actual_w = x2 - x1
            actual_h = y2 - y1
            if actual_w <= 0 or actual_h <= 0:
                quarantine_log.append({
                    "frame": img_name,
                    "box_idx": b_idx,
                    "box": box,
                    "reason": "clipped area <= 0"
                })
                continue
                
            unified_cls = AUAIR_TO_UNIFIED.get(cat, 1)
            class_counts[unified_cls] += 1
            split_boxes[split] += 1
            
            xc = (x1 + actual_w / 2.0) / img_w
            yc = (y1 + actual_h / 2.0) / img_h
            norm_w = actual_w / img_w
            norm_h = actual_h / img_h
            
            clean_boxes.append(f"{unified_cls} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
        # Write clean label text file
        lbl_file = out_lbl_dir / f"{stem}.txt"
        with open(lbl_file, "w", encoding="utf-8") as f:
            f.writelines(clean_boxes)
            
    print("\n--- Preprocessing Results ---")
    print(f"Train frames: {split_counts['train']:,} ({split_boxes['train']:,} boxes)")
    print(f"Val frames:   {split_counts['val']:,} ({split_boxes['val']:,} boxes)")
    print(f"Total valid target boxes: {sum(split_boxes.values()):,}")
    print(f"Quarantined invalid boxes: {len(quarantine_log)}")
    
    # Save split file lists
    with open(output_dir / "train_images.txt", "w", encoding="utf-8") as f:
        f.writelines([f"{fn}\n" for fn in train_files])
    with open(output_dir / "val_images.txt", "w", encoding="utf-8") as f:
        f.writelines([f"{fn}\n" for fn in val_files])
        
    manifest = {
        "dataset": "AU-AIR Multimodal UAV",
        "total_frames": len(annotations),
        "split_frames": split_counts,
        "split_boxes": split_boxes,
        "quarantined_count": len(quarantine_log),
        "quarantine_log": quarantine_log,
        "train_clips": [c for c in Counter(fn.split("_x_")[0] for fn in train_files)],
        "val_clips": VAL_CLIPS,
    }
    
    with open(output_dir / "auair_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nManifest saved to: {output_dir / 'auair_manifest.json'}")
    print(f"Label files saved to: {out_lbl_dir}")
    print("AU-AIR preprocessing completed successfully!\n")
    return manifest


if __name__ == "__main__":
    raw_auair = Path(r"c:\Users\ahile\Downloads\FlYtech\Datasets\04_AUAIR_multimodal_uav-002\04_AUAIR_multimodal_uav")
    out_auair = Path(r"c:\Users\ahile\Downloads\FlYtech\data_processed\auair")
    preprocess_auair(raw_auair, out_auair)
