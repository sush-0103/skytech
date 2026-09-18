"""
src/models/rul_predictor.py
Custom Deep Neural Architecture for Turbofan & UAV Component Health Prognostics (NASA C-MAPSS).
Predicts:
  1. Early-Warning Impending Failure Classification (RUL <= 30 cycles: CRITICAL vs NOMINAL)
  2. Continuous Remaining Useful Life (RUL) Regression (0 to 125 cycles)
Combines:
  - Multi-scale 1D Dilated Temporal Residual Convolutions
  - Bi-directional GRU Temporal Sequence Modeling
  - Multi-Head Self-Attention Pooling
Optimized for NVIDIA Tensor Core FP16 (Blackwell sm_120) and ONNX export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualBlock1D(nn.Module):
    """1D Dilated Residual Block for temporal degradation pattern extraction."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            padding=1, dilation=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + res)


class CMAPSSRULPredictor(nn.Module):
    """
    Dual-Head Predictive Maintenance Neural Network for NASA C-MAPSS.
    Zero external checkpoints; fully trained from scratch.
    """
    def __init__(
        self,
        in_features: int = 20,
        seq_len: int = 30,
        hidden_dim: int = 128,
        num_heads: int = 4
    ):
        super().__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # 1. 1D Temporal Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True)
        )

        # 2. Multi-Scale Dilated Residual Temporal Backbone
        self.res1 = TemporalResidualBlock1D(64, 64, dilation=1)
        self.res2 = TemporalResidualBlock1D(64, hidden_dim, dilation=2)
        self.res3 = TemporalResidualBlock1D(hidden_dim, hidden_dim, dilation=4)

        # 3. Bi-directional GRU Sequence Encoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 4. Multi-Head Temporal Self-Attention
        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 5. Dual Decoupled Heads
        feature_dim = hidden_dim * 2  # concat mean and max pool

        # Head A: Impending Failure Classifier (Binary: 0=NOMINAL, 1=CRITICAL_RISK)
        self.classifier_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 2)
        )

        # Head B: Continuous Remaining Useful Life Regressor (Cycles 0..125)
        self.rul_regressor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
            nn.ReLU()  # RUL cannot be negative
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (B, T, C) where T=seq_len, C=in_features
        Returns:
            logits: (B, 2) class logits for failure risk
            rul_pred: (B, 1) predicted continuous RUL cycles
        """
        # Transpose for Conv1d: (B, T, C) -> (B, C, T)
        x_conv = x.transpose(1, 2)

        # Multi-scale temporal residual convolution
        feat = self.stem(x_conv)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.res3(feat)  # (B, hidden_dim, T)

        # Transpose back for sequence & attention: (B, hidden_dim, T) -> (B, T, hidden_dim)
        seq = feat.transpose(1, 2)

        # Bi-directional GRU
        gru_out, _ = self.gru(seq)  # (B, T, hidden_dim)

        # Self-Attention
        norm_seq = self.ln_attn(gru_out)
        attn_out, _ = self.attn(norm_seq, norm_seq, norm_seq)
        combined = gru_out + attn_out  # (B, T, hidden_dim)

        # Global temporal pooling (Mean + Max concat)
        # Transpose back for 1D pooling: (B, hidden_dim, T)
        comb_t = combined.transpose(1, 2)
        avg_pool = F.adaptive_avg_pool1d(comb_t, 1).squeeze(-1)  # (B, hidden_dim)
        max_pool = F.adaptive_max_pool1d(comb_t, 1).squeeze(-1)  # (B, hidden_dim)
        pooled = torch.cat([avg_pool, max_pool], dim=-1)         # (B, hidden_dim * 2)

        # Dual head outputs
        logits = self.classifier_head(pooled)
        rul_pred = self.rul_regressor(pooled).squeeze(-1)

        return logits, rul_pred


def get_model(in_features: int = 20, seq_len: int = 30):
    return CMAPSSRULPredictor(in_features=in_features, seq_len=seq_len)


if __name__ == "__main__":
    model = get_model(in_features=20, seq_len=30)
    dummy_input = torch.randn(4, 30, 20)
    logits, rul = model(dummy_input)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[CMAPSSRULPredictor] Initialized successfully.")
    print(f"  Input shape:   {dummy_input.shape}")
    print(f"  Logits shape:  {logits.shape}")
    print(f"  RUL shape:     {rul.shape}")
    print(f"  Total params:  {param_count:,}")
