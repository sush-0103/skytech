"""
src/data/detection_dataset.py
PyTorch Dataset and DataLoader for Tactical Aerial Object Detection.
- Unifies AU-AIR and VisDrone datasets into the 10-class tactical taxonomy.
- Implements aspect-ratio-preserving letterboxing to 960x960.
- Mosaic 4-image augmentation and random horizontal flipping.
- Handles score-zero VisDrone ignore regions via binary ignore masks for loss computation.
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    class Dataset: pass
    class DataLoader: pass
    torch = None

UNIFIED_CLASSES = [
    "person", "car", "van", "truck", "bicycle",
    "motorized_two_wheeler", "bus", "trailer", "tricycle", "awning_tricycle"
]


def letterbox_image(img: Image.Image, target_size: int = 960):
    """
    Resizes image to fit inside (target_size, target_size) while preserving aspect ratio.
    Returns letterboxed image, scale ratio, pad_x, pad_y.
    """
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    
    resized_img = img.resize((new_w, new_h), Image.BILINEAR)
    
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    canvas.paste(resized_img, (pad_x, pad_y))
    
    return canvas, scale, pad_x, pad_y, w, h


class TacticalDetectionDataset(Dataset):
    def __init__(
        self,
        sources=("visdrone", "auair"),
        split="train",
        target_size=960,
        mosaic_prob=0.5,
        augment=True
    ):
        self.sources = [s.lower() for s in sources]
        self.split = split
        self.target_size = target_size
        self.mosaic_prob = mosaic_prob if split == "train" else 0.0
        self.augment = augment and (split == "train")
        
        self.samples = []
        
        # 1. VisDrone samples
        if "visdrone" in self.sources:
            vd_root = Path(r"c:\Users\ahile\Downloads\FlYtech\data_processed\visdrone")
            vd_raw = Path(r"c:\Users\ahile\Downloads\FlYtech\Datasets\aerial detection-20260918T040917Z-1-001\aerial detection\05_VisDrone_detection_tracking")
            
            vd_split = "train" if split == "train" else "val"
            img_dir = vd_raw / f"VisDrone2019-DET-{vd_split}" / "images"
            lbl_dir = vd_root / vd_split / "labels"
            ign_dir = vd_root / vd_split / "ignore_regions"
            
            if lbl_dir.exists():
                for lf in lbl_dir.glob("*.txt"):
                    stem = lf.stem
                    ip = img_dir / f"{stem}.jpg"
                    if ip.exists():
                        ig_file = ign_dir / f"{stem}.txt"
                        self.samples.append({
                            "source": "visdrone",
                            "img_path": ip,
                            "lbl_path": lf,
                            "ignore_path": ig_file if ig_file.exists() else None
                        })
                        
        # 2. AU-AIR samples
        if "auair" in self.sources:
            au_root = Path(r"c:\Users\ahile\Downloads\FlYtech\data_processed\auair")
            au_raw = Path(r"c:\Users\ahile\Downloads\FlYtech\Datasets\04_AUAIR_multimodal_uav-002\04_AUAIR_multimodal_uav\images")
            
            split_file = au_root / ("train_images.txt" if split == "train" else "val_images.txt")
            lbl_dir = au_root / "labels"
            
            if split_file.exists():
                with open(split_file, "r") as f:
                    for line in f:
                        fn = line.strip()
                        if fn:
                            stem = Path(fn).stem
                            ip = au_raw / fn
                            lp = lbl_dir / f"{stem}.txt"
                            if ip.exists() and lp.exists():
                                self.samples.append({
                                    "source": "auair",
                                    "img_path": ip,
                                    "lbl_path": lp,
                                    "ignore_path": None
                                })
                                
        print(f"TacticalDetectionDataset [{split}]: Loaded {len(self.samples):,} samples from {self.sources}")

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, idx):
        item = self.samples[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        orig_w, orig_h = img.size
        
        boxes = []
        with open(item["lbl_path"], "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append([cls_id, xc, yc, w, h])
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)
        
        # Load ignore regions if available
        ignores = []
        if item.get("ignore_path"):
            with open(item["ignore_path"], "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        ignores.append([float(x) for x in parts[:4]])  # x1, y1, w, h
        ignores = np.array(ignores, dtype=np.float32) if ignores else np.zeros((0, 4), dtype=np.float32)
        
        return img, boxes, ignores, orig_w, orig_h

    def _load_mosaic(self, idx):
        # 4-image mosaic augmentation
        indices = [idx] + [random.randint(0, len(self.samples) - 1) for _ in range(3)]
        s = self.target_size
        xc = int(random.uniform(s * 0.4, s * 0.6))
        yc = int(random.uniform(s * 0.4, s * 0.6))
        
        mosaic_img = Image.new("RGB", (s, s), (114, 114, 114))
        mosaic_boxes = []
        
        for i, index in enumerate(indices):
            img, boxes, _, w, h = self._load_sample(index)
            
            # Sub-quadrant coordinates
            if i == 0:  # Top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # Top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # Bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # Bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s), min(s, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                
            crop_w = x2b - x1b
            crop_h = y2b - y1b
            if crop_w > 0 and crop_h > 0:
                cropped = img.crop((x1b, y1b, x2b, y2b))
                mosaic_img.paste(cropped, (x1a, y1a))
                
                # Transform boxes
                pad_w = x1a - x1b
                pad_h = y1a - y1b
                if len(boxes) > 0:
                    bx1 = (boxes[:, 1] - boxes[:, 3] / 2) * w + pad_w
                    by1 = (boxes[:, 2] - boxes[:, 4] / 2) * h + pad_h
                    bx2 = (boxes[:, 1] + boxes[:, 3] / 2) * w + pad_w
                    by2 = (boxes[:, 2] + boxes[:, 4] / 2) * h + pad_h
                    
                    # Clip to quadrant
                    bx1 = np.clip(bx1, x1a, x2a)
                    by1 = np.clip(by1, y1a, y2a)
                    bx2 = np.clip(bx2, x1a, x2a)
                    by2 = np.clip(by2, y1a, y2a)
                    
                    bw = bx2 - bx1
                    bh = by2 - by1
                    valid = (bw > 3) & (bh > 3)
                    
                    if np.any(valid):
                        cls_valid = boxes[valid, 0]
                        n_xc = (bx1[valid] + bw[valid] / 2.0) / s
                        n_yc = (by1[valid] + bh[valid] / 2.0) / s
                        n_w = bw[valid] / s
                        n_h = bh[valid] / s
                        for j in range(len(cls_valid)):
                            mosaic_boxes.append([cls_valid[j], n_xc[j], n_yc[j], n_w[j], n_h[j]])
                            
        out_boxes = np.array(mosaic_boxes, dtype=np.float32) if mosaic_boxes else np.zeros((0, 5), dtype=np.float32)
        ignore_mask = np.ones((s // 4, s // 4), dtype=np.float32)
        return mosaic_img, out_boxes, ignore_mask

    def __getitem__(self, idx):
        if self.augment and random.random() < self.mosaic_prob:
            img, boxes, ignore_mask = self._load_mosaic(idx)
        else:
            raw_img, raw_boxes, raw_ignores, orig_w, orig_h = self._load_sample(idx)
            img, scale, pad_x, pad_y, w, h = letterbox_image(raw_img, self.target_size)
            
            # Transform normalized boxes to letterbox space
            s = self.target_size
            if len(raw_boxes) > 0:
                bx1 = (raw_boxes[:, 1] - raw_boxes[:, 3] / 2) * w * scale + pad_x
                by1 = (raw_boxes[:, 2] - raw_boxes[:, 4] / 2) * h * scale + pad_y
                bx2 = (raw_boxes[:, 1] + raw_boxes[:, 3] / 2) * w * scale + pad_x
                by2 = (raw_boxes[:, 2] + raw_boxes[:, 4] / 2) * h * scale + pad_y
                
                xc = ((bx1 + bx2) / 2.0) / s
                yc = ((by1 + by2) / 2.0) / s
                bw = (bx2 - bx1) / s
                bh = (by2 - by1) / s
                boxes = np.stack([raw_boxes[:, 0], xc, yc, bw, bh], axis=1).astype(np.float32)
            else:
                boxes = np.zeros((0, 5), dtype=np.float32)
                
            # Build ignore mask (stride 4)
            mask_size = s // 4
            ignore_mask = np.ones((mask_size, mask_size), dtype=np.float32)
            if len(raw_ignores) > 0:
                scale_mask = scale / 4.0
                px_mask = pad_x / 4.0
                py_mask = pad_y / 4.0
                for ign in raw_ignores:
                    ix1 = int(np.clip(ign[0] * scale_mask + px_mask, 0, mask_size))
                    iy1 = int(np.clip(ign[1] * scale_mask + py_mask, 0, mask_size))
                    ix2 = int(np.clip((ign[0] + ign[2]) * scale_mask + px_mask, 0, mask_size))
                    iy2 = int(np.clip((ign[1] + ign[3]) * scale_mask + py_mask, 0, mask_size))
                    ignore_mask[iy1:iy2, ix1:ix2] = 0.0
                    
        # Horizontal flip augmentation
        if self.augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]
            ignore_mask = np.fliplr(ignore_mask)
            
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_tensor = np.transpose(img_arr, (2, 0, 1))  # (3, 960, 960)
        
        if torch is not None:
            return (
                torch.from_numpy(img_tensor).float(),
                torch.from_numpy(boxes).float(),
                torch.from_numpy(ignore_mask.copy()).float()
            )
        return img_tensor, boxes, ignore_mask


def detection_collate_fn(batch):
    """
    Collate function to batch images, targets with batch indices, and ignore masks.
    """
    images, targets, ignore_masks = zip(*batch)
    stacked_images = torch.stack(images, 0)
    stacked_ignores = torch.stack(ignore_masks, 0)
    
    # Prepend batch index to each box: [batch_idx, class, xc, yc, w, h]
    annotated_targets = []
    for batch_idx, boxes in enumerate(targets):
        if len(boxes) > 0:
            b_indices = torch.full((boxes.shape[0], 1), batch_idx, dtype=torch.float32)
            annotated_targets.append(torch.cat([b_indices, boxes], dim=1))
            
    if annotated_targets:
        stacked_targets = torch.cat(annotated_targets, 0)
    else:
        stacked_targets = torch.zeros((0, 6), dtype=torch.float32)
        
    return stacked_images, stacked_targets, stacked_ignores


def get_detection_loaders(
    sources=("visdrone", "auair"),
    batch_size: int = 4,
    target_size: int = 960,
    num_workers: int = 0
):
    train_set = TacticalDetectionDataset(sources=sources, split="train", target_size=target_size, mosaic_prob=0.5, augment=True)
    val_set = TacticalDetectionDataset(sources=sources, split="val", target_size=target_size, mosaic_prob=0.0, augment=False)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=detection_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=detection_collate_fn, num_workers=num_workers)
    
    return train_loader, val_loader
