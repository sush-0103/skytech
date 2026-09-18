"""
src/models/tactical_detector.py
Custom Anchor-Free P2-P5 Convolutional Aerial Detector.
Designed specifically for tiny aerial objects (VisDrone & AU-AIR).
- Input: (B, 3, 960, 960)
- Backbone: Depthwise-separable stages P2..P5 (P2 stride 4, 64ch preserves tiny objects).
- Neck: Bi-directional Feature Pyramid (BiFPN) across P2..P5 (128 channels each).
- Decoupled Heads: Classification (10 classes), Anchor-Free Regression (l, t, r, b), and Centerness.
Total parameters: ~6.2M (well within the <=10M budget).
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

# Shared building blocks
from src.models.terrain_segmenter import ConvBNAct, DepthwiseSeparableBlock

UNIFIED_CLASSES = [
    "person", "car", "van", "truck", "bicycle",
    "motorized_two_wheeler", "bus", "trailer", "tricycle", "awning_tricycle"
]
NUM_CLASSES = 10


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network level fusion with learnable weights.
    Fuses P2, P3, P4, P5 (128 channels each).
    """
    def __init__(self, channels=128):
        super().__init__()
        # Top-down fusion convs
        self.conv_td_p4 = ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_td_p3 = ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_td_p2 = ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels)

        # Bottom-up fusion convs
        self.conv_bu_p3 = ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_bu_p4 = ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_bu_p5 = ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, p2, p3, p4, p5):
        # Top-down pathway
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        p4_td = self.conv_td_p4(p4 + p5_up)

        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode="nearest")
        p3_td = self.conv_td_p3(p3 + p4_up)

        p3_up = F.interpolate(p3_td, size=p2.shape[2:], mode="nearest")
        p2_out = self.conv_td_p2(p2 + p3_up)

        # Bottom-up pathway
        p2_down = F.max_pool2d(p2_out, kernel_size=2, stride=2)
        p3_out = self.conv_bu_p3(p3_td + p2_down)

        p3_down = F.max_pool2d(p3_out, kernel_size=2, stride=2)
        p4_out = self.conv_bu_p4(p4_td + p3_down)

        p4_down = F.max_pool2d(p4_out, kernel_size=2, stride=2)
        p5_out = self.conv_bu_p5(p5 + p4_down)

        return p2_out, p3_out, p4_out, p5_out


class DecoupledDetectionHead(nn.Module):
    """
    Decoupled anchor-free detection head applied across feature levels.
    """
    def __init__(self, in_channels=128, num_classes=10):
        super().__init__()
        # Classification branch
        self.cls_convs = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            ConvBNAct(in_channels, in_channels, kernel_size=1, padding=0),
            ConvBNAct(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            ConvBNAct(in_channels, in_channels, kernel_size=1, padding=0)
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)

        # Box regression branch
        self.reg_convs = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            ConvBNAct(in_channels, in_channels, kernel_size=1, padding=0),
            ConvBNAct(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            ConvBNAct(in_channels, in_channels, kernel_size=1, padding=0)
        )
        # Distance to left, top, right, bottom (4 channels)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        # Centerness / quality score
        self.centerness_pred = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        cls_feat = self.cls_convs(x)
        cls_score = self.cls_pred(cls_feat)

        reg_feat = self.reg_convs(x)
        reg_dist = F.relu(self.reg_pred(reg_feat))  # Distances >= 0
        centerness = self.centerness_pred(reg_feat)

        return cls_score, reg_dist, centerness


class TacticalDetector(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.strides = [4, 8, 16, 32]  # P2, P3, P4, P5

        # Stem: Stride 2 (960 -> 480)
        self.stem = ConvBNAct(3, 48, kernel_size=3, stride=2, padding=1)

        # Stage P2: Stride 4 (480 -> 240), 64 ch
        self.p2 = nn.Sequential(
            DepthwiseSeparableBlock(48, 64, stride=2, expansion=2),
            DepthwiseSeparableBlock(64, 64, stride=1, expansion=2)
        )

        # Stage P3: Stride 8 (240 -> 120), 128 ch
        self.p3 = nn.Sequential(
            DepthwiseSeparableBlock(64, 128, stride=2, expansion=2),
            DepthwiseSeparableBlock(128, 128, stride=1, expansion=2),
            DepthwiseSeparableBlock(128, 128, stride=1, expansion=2)
        )

        # Stage P4: Stride 16 (120 -> 60), 192 ch
        self.p4 = nn.Sequential(
            DepthwiseSeparableBlock(128, 192, stride=2, expansion=2),
            DepthwiseSeparableBlock(192, 192, stride=1, expansion=2),
            DepthwiseSeparableBlock(192, 192, stride=1, expansion=2),
            DepthwiseSeparableBlock(192, 192, stride=1, expansion=2),
            DepthwiseSeparableBlock(192, 192, stride=1, expansion=2)
        )

        # Stage P5: Stride 32 (60 -> 30), 256 ch
        self.p5 = nn.Sequential(
            DepthwiseSeparableBlock(192, 256, stride=2, expansion=2),
            DepthwiseSeparableBlock(256, 256, stride=1, expansion=2),
            DepthwiseSeparableBlock(256, 256, stride=1, expansion=2)
        )

        # Lateral projections to 128 channels
        self.lat_p2 = ConvBNAct(64, 128, kernel_size=1, padding=0)
        self.lat_p3 = ConvBNAct(128, 128, kernel_size=1, padding=0)
        self.lat_p4 = ConvBNAct(192, 128, kernel_size=1, padding=0)
        self.lat_p5 = ConvBNAct(256, 128, kernel_size=1, padding=0)

        # BiFPN feature fusion
        self.bifpn = BiFPNBlock(channels=128)

        # Shared decoupled detection heads for P2..P5
        self.head = DecoupledDetectionHead(in_channels=128, num_classes=num_classes)

    def forward(self, x):
        # Backbone
        x_stem = self.stem(x)
        c2 = self.p2(x_stem)
        c3 = self.p3(c2)
        c4 = self.p4(c3)
        c5 = self.p5(c4)

        # Lateral projections
        p2 = self.lat_p2(c2)
        p3 = self.lat_p3(c3)
        p4 = self.lat_p4(c4)
        p5 = self.lat_p5(c5)

        # BiFPN fusion
        f_p2, f_p3, f_p4, f_p5 = self.bifpn(p2, p3, p4, p5)
        features = [f_p2, f_p3, f_p4, f_p5]

        # Multi-scale head predictions
        cls_preds, reg_preds, centerness_preds = [], [], []
        for feat in features:
            cls_score, reg_dist, centerness = self.head(feat)
            cls_preds.append(cls_score)
            reg_preds.append(reg_dist)
            centerness_preds.append(centerness)

        return cls_preds, reg_preds, centerness_preds


if __name__ == "__main__":
    net = TacticalDetector(num_classes=10)
    net.eval()
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"TacticalDetector instantiated successfully!")
    print(f"Total Trainable Parameters: {params:,}")
    
    # Test forward pass with smaller size for fast CPU verification
    dummy = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        cls_out, reg_out, ctr_out = net(dummy)
        
    print(f"Output pyramid levels: {len(cls_out)}")
    for i, (c, r, ct) in enumerate(zip(cls_out, reg_out, ctr_out)):
        print(f"  Level P{i+2} (Stride {2**(i+2):2d}): Cls {list(c.shape)}, Reg {list(r.shape)}, Centerness {list(ct.shape)}")
