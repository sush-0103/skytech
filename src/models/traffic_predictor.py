"""
src/models/traffic_predictor.py
Custom Deep Neural Architecture for 4D Air Traffic Trajectory & Conflict Prediction.
Designed from scratch for UAV Detect-and-Avoid (DAA) and Cooperative Airspace Deconfliction.
- Input: History ADS-B State Sequence (B, T_in, 8)
- Backbone: 1D Dilated Temporal Residual Convolutions + Multi-Head Temporal Self-Attention
- Dual Heads:
  1. Airspace Conflict Risk Head: Predicts loss-of-separation / corridor encroachment (B, 1)
  2. 4D Trajectory Horizon Head: Forecasts future 3D displacements (Delta x, Delta y, Delta z) (B, T_out * 3)
Optimized for NVIDIA Tensor Cores (Blackwell sm_120) with static ONNX compatibility.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualBlock(nn.Module):
    """1D Dilated Residual Block for capturing multi-scale flight dynamics."""
    def __init__(self, in_channels, out_channels, dilation=1):
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

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + res)


class AirspaceTrafficPredictor(nn.Module):
    """
    Project-owned custom neural architecture for ADS-B cooperative traffic prediction.
    Zero external checkpoints or third-party pre-trained weights.
    """
    def __init__(self, in_features=8, hidden_dim=128, fut_steps=3, num_heads=4):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.fut_steps = fut_steps

        # 1. Feature Projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.SiLU(inplace=True)
        )

        # 2. Temporal Dilated Convolutions (captures acceleration and rate-of-turn trends)
        self.temp_res1 = TemporalResidualBlock(64, hidden_dim, dilation=1)
        self.temp_res2 = TemporalResidualBlock(hidden_dim, hidden_dim, dilation=2)

        # 3. Multi-Head Temporal Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # 4. Airspace Conflict Head (Classification)
        self.conflict_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 1)
        )

        # 5. Trajectory Horizon Head (3D Regression: Delta X, Delta Y, Delta Z for each future step)
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, fut_steps * 3)
        )

    def forward(self, x):
        """
        x: (B, T_in, 8) ADS-B history state tensor
        Returns:
            conflict_logits: (B, 1)
            future_trajectory: (B, fut_steps * 3)
        """
        B, T, C = x.shape
        
        # Project inputs
        h = self.input_proj(x)  # (B, T, 64)
        
        # Conv1d operates over (B, C, T)
        h_conv = h.transpose(1, 2)
        h_conv = self.temp_res1(h_conv)
        h_conv = self.temp_res2(h_conv)
        h_temp = h_conv.transpose(1, 2)  # (B, T, hidden_dim)

        # Self-Attention across temporal steps
        attn_out, _ = self.attention(h_temp, h_temp, h_temp)
        h_fused = self.attn_norm(h_temp + attn_out)

        # Temporal pooling (weighted mean of features across history window)
        h_pool = torch.mean(h_fused, dim=1)  # (B, hidden_dim)

        # Dual prediction heads
        conflict_logits = self.conflict_head(h_pool)
        traj_preds = self.trajectory_head(h_pool)

        return conflict_logits, traj_preds
