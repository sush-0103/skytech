"""
prepare_visdrone.py
Preprocessing and sanitization pipeline for VisDrone2019-DET dataset.
- Quarantines invalid zero-width / zero-height bounding boxes (3 boxes).
- Maps the 10 evaluated VisDrone categories to the unified 10-class tactical taxonomy.
- Separates category 0 (ignored regions) and category 11 (others) into ignore masks.
- Converts boxes to normalized [class, x_center, y_center, width, height].
- Exports clean label files and unified split manifests.
"""

import os
import json
from pathlib import Path
from PIL import Image

# VisDrone native class ID to unified taxonomy
# VisDrone raw: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van,
# 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others
VISDRONE_TO_UNIFIED = {
    1: 0,   # pedestrian -> person
    2: 0,   # people -> person
    3: 4,   # bicycle -> bicycle
    4: 1,   # car -> car
    5: 2,   # van -> van
    6: 3,   # truck -> truck
    7: 8,   # tricycle -> tricycle
    8: 9,   # awning-tricycle -> awning_tricycle
    9: 6,   # bus -> bus
    10: 5,  # motor -> motorized_two_wheeler
}

UNIFIED_CLASSES = [
    "person",                 # 0
    "car",                    # 1
    "van",                    # 2
    "truck",                  # 3
    "bicycle",                # 4
    "motorized_two_wheeler",  # 5
    "bus",                    # 6
    "trailer",                # 7 (AU-AIR only)
    "tricycle",               # 8
    "awning_tricycle"         # 9
]


def preprocess_visdrone(raw_dir: Path, output_dir: Path):
    print("==================================================")
    print("        VISDRONE 2019-DET PREPROCESSING          ")
    print("==================================================")
    
    splits = ["train", "val", "test-dev"]
    stats = {}
    quarantine_log = []
    
    for split in splits:
        folder_name = f"VisDrone2019-DET-{split}"
        img_dir = raw_dir / folder_name / "images"
        anno_dir = raw_dir / folder_name / "annotations"
        
        out_lbl_dir = output_dir / split / "labels"
        out_ignore_dir = output_dir / split / "ignore_regions"
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        out_ignore_dir.mkdir(parents=True, exist_ok=True)
        
        split_stats = {
            "total_images": 0,
            "total_raw_rows": 0,
            "valid_target_boxes": 0,
            "ignore_boxes": 0,
            "quarantined_zero_area": 0,
            "class_counts": {c: 0 for c in range(10)}
        }
        
        txt_files = sorted(list(anno_dir.glob("*.txt")))
        split_stats["total_images"] = len(txt_files)
        print(f"\nProcessing {split} ({len(txt_files)} images)...")
        
        for txt_file in txt_files:
            stem = txt_file.stem
            img_path = img_dir / f"{stem}.jpg"
            
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                
            clean_boxes = []
            ignore_boxes = []
            
            with open(txt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue
                    
                split_stats["total_raw_rows"] += 1
                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                score = int(parts[4])
                cat = int(parts[5])
                trunc = int(parts[6])
                occ = int(parts[7])
                
                # Check for zero width or height defect
                if w <= 0 or h <= 0:
                    split_stats["quarantined_zero_area"] += 1
                    quarantine_log.append({
                        "dataset": "VisDrone",
                        "split": split,
                        "file": txt_file.name,
                        "line": line_idx + 1,
                        "raw": line,
                        "reason": "zero or negative width/height"
                    })
                    continue
                    
                # Clip to image boundaries
                x1 = max(0.0, min(float(img_w), x))
                y1 = max(0.0, min(float(img_h), y))
                x2 = max(0.0, min(float(img_w), x + w))
                y2 = max(0.0, min(float(img_h), y + h))
                
                actual_w = x2 - x1
                actual_h = y2 - y1
                if actual_w <= 0 or actual_h <= 0:
                    split_stats["quarantined_zero_area"] += 1
                    continue
                    
                # Ignored regions / others (score 0 or category 0 or 11)
                if cat in [0, 11] or score == 0:
                    split_stats["ignore_boxes"] += 1
                    ignore_boxes.append(f"{x1:.1f} {y1:.1f} {actual_w:.1f} {actual_h:.1f}\n")
                    continue
                    
                # Valid evaluated target classes
                if cat in VISDRONE_TO_UNIFIED:
                    unified_cls = VISDRONE_TO_UNIFIED[cat]
                    split_stats["valid_target_boxes"] += 1
                    split_stats["class_counts"][unified_cls] += 1
                    
                    # Normalized center x, center y, width, height for standard training
                    xc = (x1 + actual_w / 2.0) / img_w
                    yc = (y1 + actual_h / 2.0) / img_h
                    norm_w = actual_w / img_w
                    norm_h = actual_h / img_h
                    
                    clean_boxes.append(f"{unified_cls} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    
            # Write clean normalized label file
            with open(out_lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
                f.writelines(clean_boxes)
                
            # Write ignore regions if any
            if ignore_boxes:
                with open(out_ignore_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
                    f.writelines(ignore_boxes)
                    
        stats[split] = split_stats
        print(f"  Images: {split_stats['total_images']}")
        print(f"  Raw rows: {split_stats['total_raw_rows']}")
        print(f"  Valid target boxes: {split_stats['valid_target_boxes']}")
        print(f"  Ignore boxes separated: {split_stats['ignore_boxes']}")
        print(f"  Quarantined invalid boxes: {split_stats['quarantined_zero_area']}")
        
    manifest = {
        "dataset": "VisDrone2019-DET",
        "taxonomy": {idx: name for idx, name in enumerate(UNIFIED_CLASSES)},
        "splits": stats,
        "quarantined_boxes": quarantine_log
    }
    
    with open(output_dir / "visdrone_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n==================================================")
    print(f"VisDrone preprocessing complete!")
    print(f"Quarantined {len(quarantine_log)} invalid zero-area boxes.")
    print(f"Saved manifest to {output_dir / 'visdrone_manifest.json'}")
    print("==================================================\n")
    return manifest


if __name__ == "__main__":
    raw_visdrone = Path(r"c:\Users\ahile\Downloads\FlYtech\Datasets\aerial detection-20260918T040917Z-1-001\aerial detection\05_VisDrone_detection_tracking")
    out_visdrone = Path(r"c:\Users\ahile\Downloads\FlYtech\data_processed\visdrone")
    preprocess_visdrone(raw_visdrone, out_visdrone)
