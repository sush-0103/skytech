"""
src/models/export_onnx.py
Exports trained PyTorch perception models to static-shape ONNX format for edge / TensorRT SITL deployment.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import torch
from src.models.terrain_segmenter import TerrainSegmenter


def export_terrain_segmenter(ckpt_path: Path, onnx_out: Path, input_shape=(1, 3, 512, 512)):
    print(f"Exporting TerrainSegmenter to ONNX...")
    model = TerrainSegmenter(num_classes=5)
    
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"  Loaded weights from {ckpt_path} (mIoU: {checkpoint.get('miou', 0.0)*100:.2f}%)")
        
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_out),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["semantic_logits"]
    )
    print(f"  Successfully exported static ONNX engine to: {onnx_out}")
    print(f"  Input: {input_shape} -> Output: (1, 5, 512, 512)\n")


if __name__ == "__main__":
    ckpt = Path(r"c:\Users\ahile\Downloads\FlYtech\checkpoints\terrain_segmenter_fold1_best.pt")
    out = Path(r"c:\Users\ahile\Downloads\FlYtech\checkpoints\terrain_segmenter_512.onnx")
    export_terrain_segmenter(ckpt, out)
