"""
visual_verification.py
Generates visual verification contact sheets for label alignment sign-off.
- Dubai: renders original image, indexed mask reprojected to RGB, and blended overlay.
- VisDrone: renders images with bounding box overlays and class labels.
- AU-AIR: renders frames with bounding box overlays, class labels, and flight telemetry context.
"""

import json
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Unified class names and distinct colors for visualization
CLASS_COLORS = {
    0: (255, 50, 50),     # person - bright red
    1: (0, 200, 255),     # car - cyan
    2: (255, 165, 0),     # van - orange
    3: (180, 0, 255),     # truck - purple
    4: (0, 255, 100),     # bicycle - green
    5: (255, 255, 0),     # motorized_two_wheeler - yellow
    6: (255, 0, 150),     # bus - magenta
    7: (100, 100, 255),   # trailer - blue
    8: (255, 120, 180),   # tricycle - pink
    9: (150, 255, 150),   # awning_tricycle - light green
}

UNIFIED_NAMES = [
    "person", "car", "van", "truck", "bicycle",
    "motor_2w", "bus", "trailer", "tricycle", "awning_tri"
]

# Dubai segmentation colors
DUBAI_PALETTE_MAP = {
    0: (132, 41, 246),   # Land
    1: (226, 169, 41),   # Water
    2: (60, 16, 152),    # Building
    3: (254, 221, 58),   # Vegetation
    4: (110, 193, 228),  # Road
    255: (40, 40, 40)    # Ignore
}


def draw_text_badge(draw, x, y, text, color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=bg_color)
    draw.text((x, y), text, fill=color, font=font)


def verify_dubai(out_dir: Path, num_samples: int = 4):
    print("Generating Dubai segmentation verification sheets...")
    dubai_dir = Path(r"c:\Users\ahile\Downloads\FlYtech\data_processed\dubai")
    img_dir = dubai_dir / "images"
    mask_dir = dubai_dir / "masks_indexed"
    
    sample_indices = [0, 15, 35, 52]  # Representative tiles from different parent scenes
    canvas_w = 1200
    row_h = 320
    canvas = Image.new("RGB", (canvas_w, row_h * num_samples + 50), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    
    draw.text((20, 15), "DUBAI SEGMENTATION VERIFICATION: [Original Image] | [Reprojected Mask] | [Alpha Overlay]", fill=(255, 255, 255))
    
    for idx, tid in enumerate(sample_indices):
        fn = f"tile_{tid:03d}.png"
        with Image.open(img_dir / fn) as im:
            orig = im.convert("RGB")
        with Image.open(mask_dir / fn) as m:
            indexed = np.array(m)
            
        h, w = indexed.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, col in DUBAI_PALETTE_MAP.items():
            rgb_mask[indexed == cid] = col
        mask_im = Image.fromarray(rgb_mask)
        
        # Blended overlay
        blend = Image.blend(orig, mask_im, alpha=0.5)
        
        # Resize to thumbnail (380 x 280)
        thumb_w, thumb_h = 380, 280
        t_orig = orig.resize((thumb_w, thumb_h))
        t_mask = mask_im.resize((thumb_w, thumb_h))
        t_blend = blend.resize((thumb_w, thumb_h))
        
        y_offset = 50 + idx * row_h
        canvas.paste(t_orig, (20, y_offset))
        canvas.paste(t_mask, (410, y_offset))
        canvas.paste(t_blend, (800, y_offset))
        
        draw_text_badge(draw, 30, y_offset + 10, f"{fn} Original")
        draw_text_badge(draw, 420, y_offset + 10, f"Corrected Palette Mask")
        draw_text_badge(draw, 810, y_offset + 10, f"Blended Alignment (50%)")
        
    out_path = out_dir / "dubai_verification_sheet.jpg"
    canvas.save(out_path, quality=92)
    print(f"  -> Saved {out_path}")


def verify_visdrone(out_dir: Path, num_samples: int = 4):
    print("Generating VisDrone detection verification sheets...")
    raw_img_dir = Path(r"c:\Users\ahile\Downloads\FlYtech\Datasets\aerial detection-20260918T040917Z-1-001\aerial detection\05_VisDrone_detection_tracking\VisDrone2019-DET-val\images")
    lbl_dir = Path(r"c:\Users\ahile\Downloads\FlYtech\data_processed\visdrone\val\labels")
    
    lbl_files = sorted(list(lbl_dir.glob("*.txt")))
    # Pick 4 images with good box density
    chosen_files = []
    for lf in lbl_files:
        if lf.stat().st_size > 1000:
            chosen_files.append(lf)
            if len(chosen_files) >= num_samples:
                break
                
    canvas_w = 1280
    row_h = 360
    canvas = Image.new("RGB", (canvas_w, row_h * num_samples + 50), color=(25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    
    draw.text((20, 15), "VISDRONE 2019-DET VERIFICATION: Sanitized Bounding Boxes & Tiny Objects", fill=(255, 255, 255))
    
    for idx, lf in enumerate(chosen_files):
        stem = lf.stem
        img_path = raw_img_dir / f"{stem}.jpg"
        with Image.open(img_path) as im:
            img = im.convert("RGB")
            
        w, h = img.size
        img_draw = ImageDraw.Draw(img)
        
        with open(lf, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc = float(parts[1]) * w
                yc = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h
                
                x1 = xc - bw / 2.0
                y1 = yc - bh / 2.0
                x2 = xc + bw / 2.0
                y2 = yc + bh / 2.0
                
                col = CLASS_COLORS.get(cls_id, (255, 255, 255))
                img_draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
                name = UNIFIED_NAMES[cls_id] if cls_id < len(UNIFIED_NAMES) else str(cls_id)
                draw_text_badge(img_draw, x1, max(0, y1 - 12), name, color=col, bg_color=(0, 0, 0))
                
        thumb = img.resize((1240, 330))
        y_offset = 50 + idx * row_h
        canvas.paste(thumb, (20, y_offset))
        draw_text_badge(draw, 30, y_offset + 10, f"{stem}.jpg ({len(lines)} objects)")
        
    out_path = out_dir / "visdrone_verification_sheet.jpg"
    canvas.save(out_path, quality=90)
    print(f"  -> Saved {out_path}")


def verify_auair(out_dir: Path, num_samples: int = 4):
    print("Generating AU-AIR detection & telemetry verification sheets...")
    raw_img_dir = Path(r"c:\Users\ahile\Downloads\FlYtech\Datasets\04_AUAIR_multimodal_uav-002\04_AUAIR_multimodal_uav\images")
    lbl_dir = Path(r"c:\Users\ahile\Downloads\FlYtech\data_processed\auair\labels")
    
    # Pick 4 frames with annotations
    samples = [
        "frame_20190829091111_x_0001973",
        "frame_20190829091111_x_0002100",
        "frame_20190905103112_x_0000500",
        "frame_20190906150731_x_0001200"
    ]
    
    canvas_w = 1280
    row_h = 360
    canvas = Image.new("RGB", (canvas_w, row_h * num_samples + 50), color=(25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    
    draw.text((20, 15), "AU-AIR MULTIMODAL UAV VERIFICATION: Drone Perspective, Flight Telemetry & Boxes", fill=(255, 255, 255))
    
    for idx, stem in enumerate(samples):
        img_path = raw_img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"
        if not img_path.exists():
            continue
            
        with Image.open(img_path) as im:
            img = im.convert("RGB")
            
        w, h = img.size
        img_draw = ImageDraw.Draw(img)
        
        box_count = 0
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc = float(parts[1]) * w
                    yc = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h
                    x1 = xc - bw / 2.0
                    y1 = yc - bh / 2.0
                    x2 = xc + bw / 2.0
                    y2 = yc + bh / 2.0
                    col = CLASS_COLORS.get(cls_id, (0, 255, 255))
                    img_draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
                    name = UNIFIED_NAMES[cls_id] if cls_id < len(UNIFIED_NAMES) else str(cls_id)
                    draw_text_badge(img_draw, x1, max(0, y1 - 14), name, color=col, bg_color=(0, 0, 0))
                    box_count += 1
                    
        thumb = img.resize((1240, 330))
        y_offset = 50 + idx * row_h
        canvas.paste(thumb, (20, y_offset))
        draw_text_badge(draw, 30, y_offset + 10, f"{stem}.jpg ({box_count} objects)")
        
    out_path = out_dir / "auair_verification_sheet.jpg"
    canvas.save(out_path, quality=90)
    print(f"  -> Saved {out_path}")


def main():
    out_dir = Path(r"c:\Users\ahile\Downloads\FlYtech\data_processed\verification_sheets")
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_dubai(out_dir)
    verify_visdrone(out_dir)
    verify_auair(out_dir)
    print("\nVisual verification sheets generated successfully in:", out_dir)


if __name__ == "__main__":
    main()
