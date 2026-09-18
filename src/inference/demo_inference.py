"""
src/inference/demo_inference.py
Runs perception pipeline demo on sample satellite and drone validation images,
saving visual overlay contact sheets into data_processed/inference_outputs/.
"""

import os
import sys
from pathlib import Path
from PIL import Image

# Add project root
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.inference.perception_pipeline import AerialPerceptionPipeline


def run_demo():
    print("==================================================")
    print("      RUNNING INTEGRATED PERCEPTION INFERENCE     ")
    print("==================================================")
    
    root = Path(r"c:\Users\ahile\Downloads\FlYtech")
    seg_ckpt = root / "checkpoints" / "terrain_segmenter_fold1_best.pt"
    det_ckpt = root / "checkpoints" / "tactical_detector_best.pt"
    
    out_dir = root / "data_processed" / "inference_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = AerialPerceptionPipeline(
        segmenter_ckpt=str(seg_ckpt) if seg_ckpt.exists() else None,
        detector_ckpt=str(det_ckpt) if det_ckpt.exists() else None
    )
    
    # 1. Test Strategic Terrain Segmentation on Dubai Tile
    sample_dubai = list((root / "data_processed" / "dubai" / "images").glob("*.png"))
    if sample_dubai:
        test_img_path = sample_dubai[0]
        print(f"\nProcessing Dubai Satellite Tile: {test_img_path.name}...")
        img = Image.open(test_img_path)
        seg_res = pipeline.segment_terrain(img, target_size=512)
        
        print("  Class Distribution:")
        for k, v in seg_res["class_distribution"].items():
            print(f"    - {k}: {v}%")
        print(f"  Recommended Landing Assessment: {seg_res['recommended_landing_zone']}")
        
        # Save overlay
        overlay = pipeline.render_tactical_overlay(img, seg_res, detections=[], alpha=0.4)
        out_path = out_dir / "dubai_terrain_segmented.jpg"
        overlay.save(out_path)
        print(f"  Saved visual terrain overlay -> {out_path}")
        
    # 2. Test Tactical Aerial Detection on Drone Image (AU-AIR or VisDrone)
    auair_img_dir = root / "Datasets" / "04_AUAIR_multimodal_uav-002" / "04_AUAIR_multimodal_uav" / "images"
    sample_drone = list(auair_img_dir.glob("*.jpg"))
    if not sample_drone:
        visdrone_dir = root / "data_processed" / "visdrone" / "val"
        sample_drone = list(visdrone_dir.glob("*.jpg"))
    if sample_drone:
        test_img_path = sample_drone[0]
        print(f"\nProcessing Tactical Aerial Drone Frame: {test_img_path.name}...")
        img = Image.open(test_img_path)
        
        seg_res = pipeline.segment_terrain(img, target_size=512)
        dets = pipeline.detect_objects(img, target_size=512, conf_thresh=0.20)
        
        print(f"  Detected {len(dets)} tactical aerial/ground objects:")
        for i, d in enumerate(dets[:8]):
            print(f"    [{i+1}] {d['class_name']} (conf: {d['score']:.2f}) at {[round(c, 1) for c in d['box']]}")
        if len(dets) > 8:
            print(f"    ... and {len(dets) - 8} more objects.")
            
        overlay = pipeline.render_tactical_overlay(img, seg_res, dets, alpha=0.25)
        out_path = out_dir / "visdrone_tactical_detected.jpg"
        overlay.save(out_path)
        print(f"  Saved tactical detection overlay -> {out_path}")
        
    print("\n==================================================")
    print("Inference Demonstration Complete!")
    print(f"Results saved in: {out_dir}")
    print("==================================================")


if __name__ == "__main__":
    run_demo()
