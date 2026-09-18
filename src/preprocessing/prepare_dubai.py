"""
prepare_dubai.py
Preprocessing pipeline for Dubai Aerial Segmentation dataset.
- Repairs the incorrect classes.json palette using verified publisher colors.
- Converts raw RGB masks to 8-bit single-channel class-index masks (0..4, 255 ignore).
- Maps all 72 tiles to their 8 parent satellite images to prevent spatial data leakage.
- Generates train/val cross-validation fold manifests.
"""

import os
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Exact publisher palette mapping to unified class index
COLOR_TO_CLASS = {
    (132, 41, 246): 0,    # Land (unpaved)
    (226, 169, 41): 1,    # Water
    (60, 16, 152): 2,     # Building
    (254, 221, 58): 3,    # Vegetation
    (110, 193, 228): 4,   # Road
    (0, 0, 0): 255,       # Void / Undeclared -> Ignore
    (155, 155, 155): 255  # Unlabeled -> Ignore
}

CLASS_NAMES = {
    0: "Land",
    1: "Water",
    2: "Building",
    3: "Vegetation",
    4: "Road",
    255: "Ignore"
}

# The 72 tiles grouped into their 8 parent satellite scenes (9 tiles per parent)
PARENT_CLUSTERS = {
    "parent_1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "parent_2": [10, 12, 13, 14, 15, 16, 17, 18, 19],
    "parent_3": [20, 21, 23, 24, 25, 26, 27, 28, 29],
    "parent_4": [30, 31, 32, 34, 35, 36, 37, 38, 39],
    "parent_5": [40, 41, 42, 43, 45, 46, 47, 48, 49],
    "parent_6": [50, 51, 52, 53, 54, 56, 57, 58, 59],
    "parent_7": [60, 61, 62, 63, 64, 65, 67, 68, 69],
    "parent_8": [0, 11, 22, 33, 44, 55, 66, 70, 71]
}


def convert_mask_to_indexed(rgb_mask_arr):
    h, w, _ = rgb_mask_arr.shape
    indexed = np.full((h, w), 255, dtype=np.uint8)
    
    for (r, g, b), class_id in COLOR_TO_CLASS.items():
        match = (rgb_mask_arr[:, :, 0] == r) & (rgb_mask_arr[:, :, 1] == g) & (rgb_mask_arr[:, :, 2] == b)
        indexed[match] = class_id
        
    return indexed


def preprocess_dubai(raw_dir: Path, output_dir: Path):
    print("==================================================")
    print("      DUBAI AERIAL SEGMENTATION PREPROCESSING     ")
    print("==================================================")
    
    img_dir = raw_dir / "images"
    mask_dir = raw_dir / "masks"
    
    out_img_dir = output_dir / "images"
    out_mask_dir = output_dir / "masks_indexed"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    
    pixel_counts = {c: 0 for c in [0, 1, 2, 3, 4, 255]}
    tile_metadata = []
    
    print(f"Processing 72 tiles from {raw_dir}...")
    for i in range(72):
        fn = f"tile_{i:03d}.png"
        raw_img_path = img_dir / fn
        raw_mask_path = mask_dir / fn
        
        with Image.open(raw_img_path) as img:
            rgb_img = img.convert("RGB")
            w, h = rgb_img.size
            
        with Image.open(raw_mask_path) as mask:
            rgb_mask = mask.convert("RGB")
            mask_arr = np.array(rgb_mask)
            
        indexed_mask = convert_mask_to_indexed(mask_arr)
        
        for cid in pixel_counts:
            pixel_counts[cid] += int(np.sum(indexed_mask == cid))
            
        out_mask_path = out_mask_dir / fn
        Image.fromarray(indexed_mask, mode="L").save(out_mask_path)
        
        out_img_path = out_img_dir / fn
        if not out_img_path.exists():
            rgb_img.save(out_img_path)
            
        parent_id = None
        for p, ids in PARENT_CLUSTERS.items():
            if i in ids:
                parent_id = p
                break
                
        tile_metadata.append({
            "tile_id": i,
            "filename": fn,
            "width": w,
            "height": h,
            "parent_cluster": parent_id,
            "image_path": str(out_img_path.resolve()),
            "mask_path": str(out_mask_path.resolve())
        })
        
    total_pixels = sum(pixel_counts.values())
    print("\n--- Pixel Distribution with Corrected Palette ---")
    for cid, count in pixel_counts.items():
        name = CLASS_NAMES[cid]
        pct = (count / total_pixels) * 100
        print(f"Class {cid:3d} ({name:12s}): {count:12,d} px ({pct:6.2f}%)")
        
    folds = {}
    for fold_idx, (val_parent, val_tiles) in enumerate(PARENT_CLUSTERS.items()):
        train_tiles = []
        for p, ids in PARENT_CLUSTERS.items():
            if p != val_parent:
                train_tiles.extend(ids)
        folds[f"fold_{fold_idx + 1}"] = {
            "held_out_parent": val_parent,
            "train_tile_ids": sorted(train_tiles),
            "val_tile_ids": sorted(val_tiles),
            "train_filenames": [f"tile_{tid:03d}.png" for tid in sorted(train_tiles)],
            "val_filenames": [f"tile_{tid:03d}.png" for tid in sorted(val_tiles)]
        }
        
    manifest = {
        "dataset": "Dubai Aerial Segmentation",
        "num_tiles": 72,
        "classes": CLASS_NAMES,
        "pixel_distribution": {CLASS_NAMES[cid]: pixel_counts[cid] for cid in pixel_counts},
        "tiles": tile_metadata,
        "cross_validation_folds": folds
    }
    
    manifest_path = output_dir / "dubai_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nManifest saved to: {manifest_path}")
    print(f"Indexed masks saved to: {out_mask_dir}")
    print("Dubai preprocessing completed successfully!\n")
    return manifest


if __name__ == "__main__":
    raw_dubai = Path(r"c:\Users\ahile\Downloads\FlYtech\Datasets\drone and aviation-20260918T043621Z-1-003\drone and aviation\aerial segmentation\02_Dubai_aerial_segmentation\02_Dubai_aerial_segmentation")
    out_dubai = Path(r"c:\Users\ahile\Downloads\FlYtech\data_processed\dubai")
    preprocess_dubai(raw_dubai, out_dubai)
