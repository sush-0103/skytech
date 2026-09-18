"""
src/models/terrain_segmenter.py
Custom Lightweight Aerial Segmentation Encoder-Decoder.
- Stem: 3x3 Conv, stride 2, 48 channels.
- Encoder: P2-P5 depthwise-separable residual stages (64, 128, 192, 256 channels).
- Multi-scale Context: Dilated depthwise blocks at P4 with rates (1, 2, 4, 8).
- Decoder: Skip fusion of P2, P3, P4 projected to 96 channels, bilinearly upsampled.
- Dual Heads: 5-class semantic segmentation head + auxiliary boundary head.
Total parameters: ~5.8M (within the <=7M budget).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Standard Conv2d + BatchNorm2d + SiLU activation block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableBlock(nn.Module):
    """Residual Depthwise-Separable Convolutional Block."""
    def __init__(self, in_ch, out_ch, stride=1, expansion=2):
        super().__init__()
        self.stride = stride
        hidden_dim = in_ch * expansion
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        # Expansion pointwise
        if expansion != 1:
            layers.append(ConvBNAct(in_ch, hidden_dim, kernel_size=1, padding=0))
        else:
            hidden_dim = in_ch

        # Depthwise spatial convolution
        layers.append(ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim))

        # Projection pointwise (linear without activation)
        layers.append(nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        if self.use_residual:
            return self.act(x + self.block(x))
        return self.act(self.block(x))


class DilatedContextBlock(nn.Module):
    """Multi-Scale Dilated Depthwise Context Block at P4."""
    def __init__(self, in_ch=192, out_ch=192):
        super().__init__()
        branch_ch = in_ch // 4  # 48 channels each
        
        self.b1 = ConvBNAct(in_ch, branch_ch, kernel_size=3, padding=1, dilation=1, groups=branch_ch)
        self.b2 = ConvBNAct(in_ch, branch_ch, kernel_size=3, padding=2, dilation=2, groups=branch_ch)
        self.b3 = ConvBNAct(in_ch, branch_ch, kernel_size=3, padding=4, dilation=4, groups=branch_ch)
        self.b4 = ConvBNAct(in_ch, branch_ch, kernel_size=3, padding=8, dilation=8, groups=branch_ch)
        
        self.project = ConvBNAct(branch_ch * 4, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        o1 = self.b1(x)
        o2 = self.b2(x)
        o3 = self.b3(x)
        o4 = self.b4(x)
        cat = torch.cat([o1, o2, o3, o4], dim=1)
        return self.project(cat)


class TerrainSegmenter(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        # Stem: Stride 2 (512 -> 256)
        self.stem = ConvBNAct(3, 48, kernel_size=3, stride=2, padding=1)

        # Stage P2: Stride 4 (256 -> 128), 64 ch
        self.p2 = nn.Sequential(
            DepthwiseSeparableBlock(48, 64, stride=2, expansion=2),
            DepthwiseSeparableBlock(64, 64, stride=1, expansion=2)
        )

        # Stage P3: Stride 8 (128 -> 64), 128 ch
        self.p3 = nn.Sequential(
            DepthwiseSeparableBlock(64, 128, stride=2, expansion=2),
            DepthwiseSeparableBlock(128, 128, stride=1, expansion=2),
            DepthwiseSeparableBlock(128, 128, stride=1, expansion=2)
        )

        # Stage P4: Stride 16 (64 -> 32), 192 ch
        self.p4 = nn.Sequential(
            DepthwiseSeparableBlock(128, 192, stride=2, expansion=2),
            DepthwiseSeparableBlock(192, 192, stride=1, expansion=2),
            DepthwiseSeparableBlock(192, 192, stride=1, expansion=2),
            DepthwiseSeparableBlock(192, 192, stride=1, expansion=2),
            DepthwiseSeparableBlock(192, 192, stride=1, expansion=2)
        )

        # Multi-scale Dilated Context at P4
        self.context = DilatedContextBlock(in_ch=192, out_ch=192)

        # Stage P5: Stride 32 (32 -> 16), 256 ch
        self.p5 = nn.Sequential(
            DepthwiseSeparableBlock(192, 256, stride=2, expansion=2),
            DepthwiseSeparableBlock(256, 256, stride=1, expansion=2),
            DepthwiseSeparableBlock(256, 256, stride=1, expansion=2)
        )

        # Decoder Skip Projections to 96 channels
        self.proj_p2 = ConvBNAct(64, 96, kernel_size=1, padding=0)
        self.proj_p3 = ConvBNAct(128, 96, kernel_size=1, padding=0)
        self.proj_p4 = ConvBNAct(192, 96, kernel_size=1, padding=0)
        self.proj_p5 = ConvBNAct(256, 96, kernel_size=1, padding=0)

        # Decoder Fusion: 96 * 4 = 384 channels -> 128 channels at P2 (stride 4)
        self.decoder_fusion = nn.Sequential(
            ConvBNAct(96 * 4, 128, kernel_size=3, padding=1),
            DepthwiseSeparableBlock(128, 128, stride=1, expansion=2)
        )

        # Final Upsampling to full resolution (128 -> 256 -> 512)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # Stride 2
            ConvBNAct(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # Full resolution
            ConvBNAct(64, 32, kernel_size=3, padding=1)
        )

        # Semantic Classifier Head (5 classes)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        # Auxiliary Boundary Head at Stride 4 (P2)
        self.boundary_head = nn.Sequential(
            ConvBNAct(128, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        # Input shape: (B, 3, H, W)
        h, w = x.shape[2], x.shape[3]

        # Stem & Encoder
        x_stem = self.stem(x)          # (B, 48, H/2, W/2)
        f_p2 = self.p2(x_stem)         # (B, 64, H/4, W/4)
        f_p3 = self.p3(f_p2)           # (B, 128, H/8, W/8)
        f_p4 = self.p4(f_p3)           # (B, 192, H/16, W/16)
        f_p4_ctx = self.context(f_p4)  # Multi-scale dilated context
        f_p5 = self.p5(f_p4_ctx)       # (B, 256, H/32, W/32)

        # Target feature map size at P2 (H/4, W/4)
        p2_h, p2_w = f_p2.shape[2], f_p2.shape[3]

        # Decoder projections & upsampling to P2 resolution
        dec_p2 = self.proj_p2(f_p2)
        dec_p3 = F.interpolate(self.proj_p3(f_p3), size=(p2_h, p2_w), mode="bilinear", align_corners=False)
        dec_p4 = F.interpolate(self.proj_p4(f_p4_ctx), size=(p2_h, p2_w), mode="bilinear", align_corners=False)
        dec_p5 = F.interpolate(self.proj_p5(f_p5), size=(p2_h, p2_w), mode="bilinear", align_corners=False)

        # Concat multi-level features
        fused = torch.cat([dec_p2, dec_p3, dec_p4, dec_p5], dim=1)  # (B, 384, H/4, W/4)
        fused_feat = self.decoder_fusion(fused)                     # (B, 128, H/4, W/4)

        # Auxiliary boundary prediction (Stride 4)
        boundary_logits = self.boundary_head(fused_feat)            # (B, 1, H/4, W/4)

        # Main semantic prediction upsampled to (H, W)
        upsampled = self.upsample(fused_feat)                       # (B, 32, H, W)
        semantic_logits = self.classifier(upsampled)                # (B, 5, H, W)

        if self.training:
            return semantic_logits, boundary_logits
        return semantic_logits


def get_parameter_count(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = TerrainSegmenter(num_classes=5)
    net.eval()
    dummy = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = net(dummy)
    print(f"TerrainSegmenter instantiated successfully!")
    print(f"Total Trainable Parameters: {get_parameter_count(net):,}")
    print(f"Output Logits Shape: {out.shape} (Expected: [2, 5, 512, 512])")
