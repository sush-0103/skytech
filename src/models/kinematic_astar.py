"""
src/models/kinematic_astar.py
Neural Kinodynamic A* Planner Architecture for 3D Autonomous UAV Navigation.
- Input: 3D Costmap / Hazard Map (B, 3, H, W) + Drone State (B, 6) + Goal (B, 3)
- Multi-scale Spatial Residual Feature Extractor with FiLM State-Goal Conditioning.
- Three Predictive Heads:
  1. Heuristic Cost-To-Go Field h(s, s_goal) (B, 1, H, W)
  2. Kinodynamic Motion Primitive Selection Policy (B, 27)
  3. Trajectory Feasibility / Clearance Logits (B, 1)
Optimized for NVIDIA Tensor Cores (Blackwell sm_120) with static ONNX compatibility.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PRIMITIVES = 27


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class KinematicAStarNet(nn.Module):
    """
    Neural Guidance Network for 3D Kinodynamic A* Search.
    Fuses spatial perception with vehicle kinodynamics to prune unpromising expansions.
    """
    def __init__(self, in_channels=3, base_channels=32, num_primitives=NUM_PRIMITIVES):
        super().__init__()
        self.num_primitives = num_primitives
        self.base_channels = base_channels
        
        # State-Goal Conditioning MLP (pos_xyz + vel_xyz + goal_xyz = 9 dims)
        self.cond_mlp = nn.Sequential(
            nn.Linear(9, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, base_channels * 8)
        )

        # Spatial Costmap Encoder (P1 to P4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True)
        )
        self.stage1 = ConvBlock(base_channels, base_channels * 2, stride=2)     # 1/2
        self.stage2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)   # 1/4
        self.stage3 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)   # 1/8
        self.stage4 = ConvBlock(base_channels * 8, base_channels * 8, stride=2)   # 1/16

        # Decoder for Dense Heuristic Map
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)

        # 1. Dense Heuristic Head: Cost-to-Go Field h(s, s_goal)
        self.heuristic_head = nn.Sequential(
            nn.Conv2d(base_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Softplus()
        )

        # 2. Motion Primitive Policy Head: Action Selection
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.primitive_head = nn.Sequential(
            nn.Linear(base_channels * 8 + base_channels * 8, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_primitives)
        )

        # 3. Trajectory Feasibility / Safety Head (Raw Logits for AMP stability)
        self.safety_head = nn.Sequential(
            nn.Linear(base_channels * 8 + base_channels * 8, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, costmap, state, goal):
        """
        costmap: (B, 3, H, W)
        state:   (B, 6) [x, y, z, vx, vy, vz]
        goal:    (B, 3) [gx, gy, gz]
        """
        sg = torch.cat([state, goal], dim=-1)
        cond_embed = self.cond_mlp(sg)

        x0 = self.stem(costmap)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)

        gamma, beta = cond_embed.chunk(2, dim=-1)
        x2_mod = x2 * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)

        x3 = self.stage3(x2_mod)
        x4 = self.stage4(x3)

        d3 = self.up4(x4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up3(d3)
        d2 = self.dec2(torch.cat([d2, x2_mod], dim=1))

        d1 = self.up2(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        d0 = self.up1(d1)
        heuristic_map = self.heuristic_head(d0)

        global_feat = self.pool(x4).flatten(1)
        policy_input = torch.cat([global_feat, cond_embed], dim=-1)

        primitive_logits = self.primitive_head(policy_input)
        safety_logits = self.safety_head(policy_input)

        return heuristic_map, primitive_logits, safety_logits
