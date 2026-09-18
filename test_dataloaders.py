"""
test_dataloaders.py
Smoke test script to verify PyTorch DataLoaders for both:
1. TerrainSegmentationDataset (Dubai 512x512 with 8-fold cross-validation)
2. TacticalDetectionDataset (VisDrone & AU-AIR with letterboxing and mosaic)
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

def test_segmentation_loader():
    print("\n--- Testing TerrainSegmentationDataset & DataLoader ---")
    from src.data.segmentation_dataset import TerrainSegmentationDataset, get_segmentation_loaders
    
    train_loader, val_loader = get_segmentation_loaders(batch_size=2, fold=1, crop_size=512)
    print(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")
    
    for images, masks in train_loader:
        print(f"Sample Batch -> Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"             -> Masks shape:  {masks.shape}, dtype: {masks.dtype}")
        print(f"             -> Unique mask values: {masks.unique().tolist()}")
        assert images.shape == (2, 3, 512, 512)
        assert masks.shape == (2, 512, 512)
        break
    print("TerrainSegmentationDataset test PASSED!")


def test_detection_loader():
    print("\n--- Testing TacticalDetectionDataset & DataLoader ---")
    from src.data.detection_dataset import TacticalDetectionDataset, get_detection_loaders
    
    # Test with VisDrone first for fast check
    train_loader, val_loader = get_detection_loaders(sources=["visdrone"], batch_size=2, target_size=960)
    print(f"VisDrone Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    for images, targets, ignores in train_loader:
        print(f"Sample Batch -> Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"             -> Targets shape (batch_idx, cls, xc, yc, w, h): {targets.shape}")
        print(f"             -> Ignore masks shape: {ignores.shape}")
        assert images.shape == (2, 3, 960, 960)
        assert ignores.shape == (2, 240, 240)
        break
    print("TacticalDetectionDataset test PASSED!")


if __name__ == "__main__":
    test_segmentation_loader()
    test_detection_loader()
    print("\nALL DATALOADER SMOKE TESTS PASSED!")
