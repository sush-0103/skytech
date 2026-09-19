#!/usr/bin/env python3
"""
tools/monitor_ai.py
Live Real-Time AI Telemetry Monitor for Terminal (CMD / PowerShell).
Streams active 3D Kinematic primitives, latency, and GPU metrics directly from port 5001.
"""

import sys
import time
import json
import urllib.request

URL = "http://127.0.0.1:5001/api/ai/live"

def main():
    print("=" * 80)
    print("  SKYTECH REAL-TIME AI INFERENCE MONITOR (NVIDIA RTX 5070)")
    print("  Connecting to http://127.0.0.1:5001/api/ai/live ... (Press Ctrl+C to stop)")
    print("=" * 80)
    print(f"{'TIME':<12} | {'ACTIVE PRIMITIVE':<22} | {'LATENCY':<10} | {'EVASION ACTION':<24} | {'GPU UTIL'}")
    print("-" * 80)

    while True:
        try:
            req = urllib.request.urlopen(URL, timeout=2.0)
            data = json.loads(req.read().decode("utf-8"))
            
            t_str = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
            prim = data["models"]["kinematic_astar"]["active_primitive"]
            lat = f"{data['models']['kinematic_astar']['latency_ms']} ms"
            action = data["reactive_avoidance"]["optimal_action"]
            gpu = f"{data['hardware']['gpu_utilization_pct']}% (VRAM: {data['hardware']['vram_used_mb']}MB)"

            print(f"{t_str:<12} | {prim:<22} | {lat:<10} | {action:<24} | {gpu}")
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[!] Monitor stopped.")
            break
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Connecting to AI server... ({e})")
            time.sleep(1.0)

if __name__ == "__main__":
    main()
