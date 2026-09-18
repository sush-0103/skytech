"""
prepare_opensky.py
Preprocessing pipeline for OpenSky ADS-B flight telemetry.
- Rejects stale position reports (age = snapshot_time - time_position > 15 s).
- Filters out null positions and on-ground aircraft.
- Reconstructs 4D aircraft state trajectories (time, lat, lon, baro_alt, velocity, heading, vertical_rate).
- Formats cooperative traffic tracks for SITL simulated airspace deconfliction.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict


def preprocess_opensky(raw_csv_path: Path, output_dir: Path, max_age_sec: float = 15.0):
    print("==================================================")
    print("       OPENSKY ADS-B TELEMETRY PREPROCESSING      ")
    print("==================================================")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_rows = 0
    rejected_stale = 0
    rejected_missing_pos = 0
    rejected_on_ground = 0
    valid_rows = 0
    
    # Store trajectories by aircraft ICAO24
    trajectories = defaultdict(list)
    snapshots = set()
    
    with open(raw_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            
            # Check snapshot timestamp
            try:
                snap_time = float(row["snapshot_time"])
                snapshots.add(snap_time)
            except (ValueError, KeyError):
                continue
                
            # Filter missing lat/lon
            lat_str = row.get("latitude", "")
            lon_str = row.get("longitude", "")
            if not lat_str or not lon_str:
                rejected_missing_pos += 1
                continue
            lat = float(lat_str)
            lon = float(lon_str)
            
            # Filter stale position (age > 15 s per audit specification)
            time_pos_str = row.get("time_position", "")
            if time_pos_str:
                pos_age = snap_time - float(time_pos_str)
                if pos_age > max_age_sec:
                    rejected_stale += 1
                    continue
            else:
                # If no time_position, check last_contact
                last_contact_str = row.get("last_contact", "")
                if last_contact_str:
                    contact_age = snap_time - float(last_contact_str)
                    if contact_age > max_age_sec:
                        rejected_stale += 1
                        continue
                        
            # Filter on-ground aircraft if specified
            on_ground = row.get("on_ground", "False").lower() == "true"
            if on_ground:
                rejected_on_ground += 1
                continue
                
            icao = row.get("icao24", "").strip().lower()
            callsign = row.get("callsign", "").strip()
            
            # Parse altitude, velocity, track
            try:
                baro_alt = float(row.get("baro_altitude") or 0.0)
            except ValueError:
                baro_alt = 0.0
                
            try:
                geo_alt = float(row.get("geo_altitude") or baro_alt)
            except ValueError:
                geo_alt = baro_alt
                
            try:
                velocity = float(row.get("velocity") or 0.0)
            except ValueError:
                velocity = 0.0
                
            try:
                true_track = float(row.get("true_track") or 0.0)
            except ValueError:
                true_track = 0.0
                
            try:
                vertical_rate = float(row.get("vertical_rate") or 0.0)
            except ValueError:
                vertical_rate = 0.0
                
            point = {
                "time": snap_time,
                "lat": lat,
                "lon": lon,
                "baro_altitude_m": baro_alt,
                "geo_altitude_m": geo_alt,
                "velocity_mps": velocity,
                "true_track_deg": true_track,
                "vertical_rate_mps": vertical_rate,
                "callsign": callsign
            }
            trajectories[icao].append(point)
            valid_rows += 1
            
    # Sort points chronologically per aircraft
    clean_trajectories = {}
    multi_point_trajectories = 0
    for icao, pts in trajectories.items():
        sorted_pts = sorted(pts, key=lambda x: x["time"])
        clean_trajectories[icao] = sorted_pts
        if len(sorted_pts) > 1:
            multi_point_trajectories += 1
            
    print(f"\n--- OpenSky Filtering Results ---")
    print(f"Total raw records:            {total_rows:,}")
    print(f"Rejected stale (age > {max_age_sec}s): {rejected_stale:,}")
    print(f"Rejected missing coordinates: {rejected_missing_pos:,}")
    print(f"Rejected on ground:           {rejected_on_ground:,}")
    print(f"Valid airborne records:       {valid_rows:,}")
    print(f"Unique aircraft tracks:       {len(clean_trajectories):,}")
    print(f"Multi-step trajectories (>1): {multi_point_trajectories:,}")
    
    # Save formatted trajectories for SITL deconfliction replay
    out_json = output_dir / "cooperative_traffic_tracks.json"
    manifest = {
        "dataset": "OpenSky Flight Telemetry",
        "description": "Cleaned ADS-B cooperative traffic trajectories for SITL airspace deconfliction",
        "total_records": total_rows,
        "valid_records": valid_rows,
        "rejected_stale": rejected_stale,
        "aircraft_count": len(clean_trajectories),
        "multi_step_count": multi_point_trajectories,
        "trajectories": clean_trajectories
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nExported cooperative tracks to: {out_json}")
    print("OpenSky preprocessing completed successfully!\n")
    return manifest


if __name__ == "__main__":
    raw_path = Path(r"c:\Users\ahile\Downloads\FlYtech\Datasets\flight telemetry-20260918T051109Z-1-001\flight telemetry\03_OpenSky_flight_telemetry\03_OpenSky_flight_telemetry\opensky_trajectories.csv")
    out_dir = Path(r"c:\Users\ahile\Downloads\FlYtech\data_processed\opensky")
    preprocess_opensky(raw_path, out_dir)
