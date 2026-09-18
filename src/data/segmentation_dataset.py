"""
src/data/segmentation_dataset.py
PyTorch Dataset and DataLoader for Dubai Aerial Semantic Segmentation.
- Loads 8-bit indexed masks (0..4 classes, 255 ignore).
- Implements 8-fold leave-one-parent-scene-out cross-validation from dubai_manifest.json.
- Extracts 512x512 patches with random flips, 90-degree rotations, and color jitter.
"""

import json
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    # Graceful fallback if torch is still downloading in background
    class Dataset: pass
    class DataLoader: pass
    torch = None

CLASS_NAMES = ["Land", "Water", "Building", "Vegetation", "Road"]
NUM_CLASSES = 5
IGNORE_INDEX = 255


class TerrainSegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: str = r"c:\Users\ahile\Downloads\FlYtech\data_processed\dubai",
        fold: int = 1,
        split: str = "train",
        crop_size: int = 512,
        augment: bool = True
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == "train")
        
        manifest_file = self.data_root / "dubai_manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        fold_key = f"fold_{fold}"
        if fold_key not in manifest["cross_validation_folds"]:
            raise ValueError(f"Fold {fold} not found in manifest. Available folds: 1..8")
            
        fold_info = manifest["cross_validation_folds"][fold_key]
        self.filenames = fold_info["train_filenames"] if split == "train" else fold_info["val_filenames"]
        
        self.img_dir = self.data_root / "images"
        self.mask_dir = self.data_root / "masks_indexed"
        
        # Pre-load all images and masks into RAM for zero-disk-IO fast training
        self.cache = {}
        for fn in self.filenames:
            with Image.open(self.img_dir / fn) as im:
                img_mem = im.convert("RGB")
            with Image.open(self.mask_dir / fn) as ma:
                mask_mem = ma.convert("L")
            self.cache[fn] = (img_mem, mask_mem)
            
    def __len__(self):
        # In training, sample 4 crops per tile per epoch (fast & diverse)
        return len(self.filenames) * (4 if self.split == "train" else 2)

    def _apply_augmentations(self, img: Image.Image, mask: Image.Image):
        # 1. Random Horizontal Flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
        # 2. Random Vertical Flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            
        # 3. Random 90/180/270 degree rotation
        rot_choice = random.choice([0, 90, 180, 270])
        if rot_choice == 90:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        elif rot_choice == 180:
            img = img.transpose(Image.ROTATE_180)
            mask = mask.transpose(Image.ROTATE_180)
        elif rot_choice == 270:
            img = img.transpose(Image.ROTATE_270)
            mask = mask.transpose(Image.ROTATE_270)
            
        # 4. Color Jitter on image only
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
            
        return img, mask

    def _extract_crop(self, img: Image.Image, mask: Image.Image):
        w, h = img.size
        cw, ch = self.crop_size, self.crop_size
        
        if w < cw or h < ch:
            # Pad if tile is smaller than crop_size
            pad_w = max(0, cw - w)
            pad_h = max(0, ch - h)
            new_img = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
            new_mask = Image.new("L", (w + pad_w, h + pad_h), IGNORE_INDEX)
            new_img.paste(img, (0, 0))
            new_mask.paste(mask, (0, 0))
            img, mask = new_img, new_mask
            w, h = img.size

        if self.split == "train":
            x1 = random.randint(0, w - cw)
            y1 = random.randint(0, h - ch)
        else:
            x1 = (w - cw) // 2
            y1 = (h - ch) // 2
            
        img_crop = img.crop((x1, y1, x1 + cw, y1 + ch))
        mask_crop = mask.crop((x1, y1, x1 + cw, y1 + ch))
        return img_crop, mask_crop

    def __getitem__(self, idx):
        file_idx = idx % len(self.filenames)
        fn = self.filenames[file_idx]
        
        # Zero-disk I/O from RAM cache
        cached_img, cached_mask = self.cache[fn]
        img = cached_img.copy()
        mask = cached_mask.copy()
        
        img_crop, mask_crop = self._extract_crop(img, mask)
        
        if self.augment:
            img_crop, mask_crop = self._apply_augmentations(img_crop, mask_crop)
            
        img_arr = np.array(img_crop, dtype=np.float32) / 255.0  # (H, W, 3)
        # Normalize with standard mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_arr = (img_arr - mean) / std
        img_tensor = np.transpose(img_arr, (2, 0, 1))  # (3, H, W)
        
        mask_arr = np.array(mask_crop, dtype=np.int64)  # (H, W) with values 0..4 or 255
        
        if torch is not None:
            return torch.from_numpy(img_tensor).float(), torch.from_numpy(mask_arr).long()
        return img_tensor, mask_arr


def get_segmentation_loaders(
    data_root: str = r"c:\Users\ahile\Downloads\FlYtech\data_processed\dubai",
    fold: int = 1,
    batch_size: int = 8,
    crop_size: int = 512,
    num_workers: int = 0
):
    train_set = TerrainSegmentationDataset(data_root=data_root, fold=fold, split="train", crop_size=crop_size, augment=True)
    val_set = TerrainSegmentationDataset(data_root=data_root, fold=fold, split="val", crop_size=crop_size, augment=False)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader
