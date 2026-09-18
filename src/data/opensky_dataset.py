"""
src/data/opensky_dataset.py
PyTorch Dataset and DataLoader for OpenSky ADS-B Cooperative Traffic Telemetry.
- Leak-free split by aircraft ICAO24 (prevents temporal data leakage between train and val).
- Standardizes local coordinates (km relative to window origin) and velocities.
- Generates dual ground-truth targets:
  1. Conflict Classification: Loss-of-separation / terminal airspace corridor encroachment.
  2. Trajectory Forecasting: (Delta x, Delta y, Delta z) future 3D displacement vectors.
"""

import json
import math
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

R_EARTH = 6371.0  # Earth radius in km

# Major regional airport / terminal airspace corridor hubs
CORRIDOR_HUBS = [
    (19.09, 72.87),  # Mumbai (VABB)
    (28.55, 77.10),  # Delhi (VIDP)
    (23.07, 72.63),  # Ahmedabad (VAAH)
    (22.65, 88.45),  # Kolkata (VECC)
    (21.09, 79.05),  # Nagpur (VANP)
    (13.20, 77.71),  # Bengaluru (VOBL)
    (17.24, 78.43)   # Hyderabad (VOHS)
]


def geo_to_local_km(lat, lon, ref_lat, ref_lon):
    """Equirectangular projection from WGS84 lat/lon to local Cartesian km."""
    d_lat = math.radians(lat - ref_lat)
    d_lon = math.radians(lon - ref_lon)
    ref_lat_rad = math.radians(ref_lat)
    y = d_lat * R_EARTH
    x = d_lon * R_EARTH * math.cos(ref_lat_rad)
    return x, y


def is_airspace_conflict(lat, lon, alt_m, vz_mps):
    """
    Evaluates whether an aircraft enters the protected terminal / autonomous corridor.
    Conflict condition: within 140km of major corridor hub below 10,500m,
    or within 90km at any altitude, or low-altitude flight below 5,500m.
    """
    min_dist_km = min(
        math.hypot((lat - hlat) * 111.0, (lon - hlon) * 111.0 * math.cos(math.radians(hlat)))
        for hlat, hlon in CORRIDOR_HUBS
    )
    return (min_dist_km < 140.0 and alt_m < 10500.0) or (min_dist_km < 90.0) or (alt_m < 5500.0)


class OpenSkyTrafficDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        split: str = "train",
        train_ratio: float = 0.80,
        hist_steps: int = 4,
        fut_steps: int = 3,
        seed: int = 42
    ):
        super().__init__()
        self.hist_steps = hist_steps
        self.fut_steps = fut_steps

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trajectories = data.get("trajectories", {})
        
        # Sort aircraft icao24 keys deterministically and split by aircraft
        all_icaos = sorted([k for k, v in trajectories.items() if len(v) >= (hist_steps + fut_steps)])
        rng = np.random.RandomState(seed)
        shuffled_icaos = rng.permutation(all_icaos)
        split_idx = int(len(shuffled_icaos) * train_ratio)

        if split == "train":
            selected_icaos = set(shuffled_icaos[:split_idx])
        else:
            selected_icaos = set(shuffled_icaos[split_idx:])

        self.samples = []

        for icao in selected_icaos:
            pts = trajectories[icao]
            total_pts = len(pts)
            if total_pts < (hist_steps + fut_steps):
                continue

            ref_lat = pts[0]["lat"]
            ref_lon = pts[0]["lon"]

            for i in range(total_pts - hist_steps - fut_steps + 1):
                hist_pts = pts[i : i + hist_steps]
                fut_pts = pts[i + hist_steps : i + hist_steps + fut_steps]

                # Convert history points to normalized local state features
                hist_feats = []
                x0, y0 = geo_to_local_km(hist_pts[0]["lat"], hist_pts[0]["lon"], ref_lat, ref_lon)
                
                for p in hist_pts:
                    x, y = geo_to_local_km(p["lat"], p["lon"], ref_lat, ref_lon)
                    dx = (x - x0) / 10.0
                    dy = (y - y0) / 10.0
                    z = (p["baro_altitude_m"] / 1000.0) / 10.0  # Normalized (10 km = 1.0)
                    
                    vel = p["velocity_mps"] / 100.0             # Normalized (100 m/s = 1.0)
                    track_rad = math.radians(p["true_track_deg"])
                    vx = vel * math.sin(track_rad)
                    vy = vel * math.cos(track_rad)
                    vz = (p["vertical_rate_mps"] or 0.0) / 10.0 # Normalized (10 m/s = 1.0)
                    
                    hist_feats.append([
                        dx, dy, z,
                        vx, vy, vz,
                        math.sin(track_rad), math.cos(track_rad)
                    ])

                # Future target offsets relative to last history point
                last_p = hist_pts[-1]
                last_x, last_y = geo_to_local_km(last_p["lat"], last_p["lon"], ref_lat, ref_lon)
                last_z = last_p["baro_altitude_m"] / 1000.0

                fut_offsets = []
                conflict = 0.0

                for fp in fut_pts:
                    fx, fy = geo_to_local_km(fp["lat"], fp["lon"], ref_lat, ref_lon)
                    fz = fp["baro_altitude_m"] / 1000.0
                    
                    fut_offsets.extend([
                        (fx - last_x) / 10.0,
                        (fy - last_y) / 10.0,
                        (fz - last_z) / 10.0
                    ])

                    # Check separation conflict against protected corridor volume
                    if is_airspace_conflict(fp["lat"], fp["lon"], fp["baro_altitude_m"], fp["vertical_rate_mps"]):
                        conflict = 1.0

                self.samples.append((
                    np.array(hist_feats, dtype=np.float32),
                    np.float32(conflict),
                    np.array(fut_offsets, dtype=np.float32)
                ))

        conflicts = sum(1 for s in self.samples if s[1] > 0.5)
        total = len(self.samples)
        pct = (conflicts / total * 100.0) if total > 0 else 0.0
        print(f"[{split.upper()}] Aircraft: {len(selected_icaos)} | Windows: {total} | Conflicts: {conflicts} ({pct:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist_feats, conflict, fut_offsets = self.samples[idx]
        return (
            torch.from_numpy(hist_feats),
            torch.tensor(conflict, dtype=torch.float32),
            torch.from_numpy(fut_offsets)
        )


def get_opensky_loaders(
    json_path: str = "data_processed/opensky/cooperative_traffic_tracks.json",
    batch_size: int = 32,
    hist_steps: int = 4,
    fut_steps: int = 3,
    seed: int = 42
):
    train_ds = OpenSkyTrafficDataset(
        json_path, split="train", train_ratio=0.80,
        hist_steps=hist_steps, fut_steps=fut_steps, seed=seed
    )
    val_ds = OpenSkyTrafficDataset(
        json_path, split="val", train_ratio=0.80,
        hist_steps=hist_steps, fut_steps=fut_steps, seed=seed
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
