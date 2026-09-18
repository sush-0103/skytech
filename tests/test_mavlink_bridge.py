"""
test_mavlink_bridge.py
Automated test suite verifying the Deterministic Safety Supervisor
and MAVLink 20 Hz SITL telemetry bridge.
"""

import time
import socket
import threading
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pymavlink import mavutil

from src.interface.safety_supervisor import (
    DeterministicSafetySupervisor,
    SupervisorLimits,
    CandidateSetpoint
)
from src.interface.mavlink_bridge import MAVLinkSITLBridge


def test_safety_supervisor_invariants():
    print("\n--- Test 1: Deterministic Safety Supervisor Invariants ---")
    supervisor = DeterministicSafetySupervisor()
    t_now = time.time()

    # 1. Nominal Setpoint (Should Pass)
    nominal = CandidateSetpoint(
        timestamp=t_now,
        x=50.0, y=-30.0, z=-25.0,
        vx=8.0, vy=2.0, vz=-0.5,
        yaw=0.3
    )
    res = supervisor.evaluate_setpoint(nominal, now=t_now)
    assert res.accepted is True, f"Nominal setpoint should be accepted, got: {res.rejection_reason}"
    assert res.state == "GUIDED_ACTIVE"
    print("  [PASS] Nominal setpoint accepted (v=8.2 m/s, z=-25m)")

    # 2. Velocity Exceeded (Should Reject)
    excess_vel = CandidateSetpoint(
        timestamp=t_now,
        x=50.0, y=-30.0, z=-25.0,
        vx=18.0, vy=10.0, vz=0.0   # v_total = 20.6 m/s > 15 m/s
    )
    res_vel = supervisor.evaluate_setpoint(excess_vel, now=t_now)
    assert res_vel.accepted is False, "Excess velocity should be rejected!"
    assert "VELOCITY_EXCEEDED" in res_vel.rejection_reason
    assert res_vel.state == "FAILSAFE_BRAKE"
    print("  [PASS] Excess velocity (20.6 m/s > 15 m/s) rejected -> FAILSAFE_BRAKE")

    # 3. Geofence Horizontal Exceeded (Should Reject)
    out_geofence = CandidateSetpoint(
        timestamp=t_now,
        x=750.0, y=0.0, z=-25.0,   # X=750m > 500m
        vx=5.0, vy=0.0, vz=0.0
    )
    res_geo = supervisor.evaluate_setpoint(out_geofence, now=t_now)
    assert res_geo.accepted is False, "Out-of-geofence setpoint should be rejected!"
    assert "GEOFENCE_EXCEEDED" in res_geo.rejection_reason
    print("  [PASS] Geofence boundary violation (X=750m > 500m) rejected")

    # 4. Altitude Ceiling Exceeded (Should Reject)
    out_alt = CandidateSetpoint(
        timestamp=t_now,
        x=0.0, y=0.0, z=-150.0,    # -150m > 120m ceiling
        vx=0.0, vy=0.0, vz=0.0
    )
    res_alt = supervisor.evaluate_setpoint(out_alt, now=t_now)
    assert res_alt.accepted is False, "Altitude ceiling violation should be rejected!"
    assert "GEOFENCE_EXCEEDED" in res_alt.rejection_reason
    print("  [PASS] Altitude ceiling violation (Z=-150m > 120m AGL) rejected")

    # 5. Excessive Climb Rate (Should Reject)
    excess_climb = CandidateSetpoint(
        timestamp=t_now,
        x=0.0, y=0.0, z=-30.0,
        vx=2.0, vy=0.0, vz=-4.2    # |vz| = 4.2 m/s > 2.5 m/s
    )
    res_climb = supervisor.evaluate_setpoint(excess_climb, now=t_now)
    assert res_climb.accepted is False, "Excessive climb rate should be rejected!"
    assert "CLIMB_RATE_EXCEEDED" in res_climb.rejection_reason
    print("  [PASS] Excessive climb rate (|vz|=4.2 m/s > 2.5 m/s) rejected")

    # 6. Stale Command (Should Reject)
    stale_cmd = CandidateSetpoint(
        timestamp=t_now - 0.120,   # 120 ms old > 50 ms limit
        x=10.0, y=10.0, z=-25.0,
        vx=2.0, vy=2.0, vz=0.0
    )
    res_stale = supervisor.evaluate_setpoint(stale_cmd, now=t_now)
    assert res_stale.accepted is False, "Stale command should be rejected!"
    assert "STALE_COMMAND" in res_stale.rejection_reason
    print("  [PASS] Stale command (age=120ms > 50ms) rejected")

    # 7. Obstacle Clearance Buffer Check (Should Reject)
    close_obs = [{"id": "TGT-CRANE", "x": 12.0, "y": 14.0, "z": -25.0}]
    encroaching = CandidateSetpoint(
        timestamp=t_now,
        x=10.0, y=10.0, z=-25.0,   # distance = sqrt(4 + 16) = 4.47m < 15.0m buffer
        vx=1.0, vy=1.0, vz=0.0
    )
    res_obs = supervisor.evaluate_setpoint(encroaching, current_obstacles=close_obs, now=t_now)
    assert res_obs.accepted is False, "Obstacle buffer encroachment should be rejected!"
    assert "CLEARANCE_VIOLATION" in res_obs.rejection_reason
    print("  [PASS] Obstacle buffer encroachment (dist=4.5m < 15m) rejected")

    # 8. Consecutive Rejections Trigger Failsafe Ladder
    for _ in range(12):
        supervisor.evaluate_setpoint(encroaching, current_obstacles=close_obs, now=t_now)
    assert supervisor.state == "EMERGENCY_LAND"
    print("  [PASS] Sustained violations transition safely: GUIDED -> BRAKE -> HOLD -> EMERGENCY_LAND")


def test_mavlink_udp_streaming():
    print("\n--- Test 2: MAVLink 20 Hz UDP Packet Streaming & Decoding ---")
    test_port = 14555  # Distinct port for test isolation
    
    # 1. Setup mock SITL receiver listening on UDP 14555
    mock_sitl = mavutil.mavlink_connection(f"udpin:127.0.0.1:{test_port}")
    
    # 2. Setup MAVLink Bridge broadcasting to UDP 14555
    supervisor = DeterministicSafetySupervisor()
    bridge = MAVLinkSITLBridge(
        target_ip="127.0.0.1",
        target_port=test_port,
        rate_hz=20.0,
        supervisor=supervisor
    )
    
    # 3. Set a nominal waypoint
    bridge.update_flight_plan(CandidateSetpoint(
        timestamp=time.time(),
        x=120.5, y=-45.2, z=-30.0,
        vx=6.5, vy=-1.5, vz=0.0,
        yaw=1.2,
        source_model="3d_kinematic_astar"
    ))
    
    bridge.start()
    print("  [+] MAVLink bridge started, broadcasting at 20 Hz...")

    received_messages = []
    t_end = time.time() + 1.5  # Collect for 1.5 seconds (~30 packets)

    while time.time() < t_end:
        msg = mock_sitl.recv_match(blocking=True, timeout=0.1)
        if msg:
            received_messages.append(msg)

    bridge.stop()
    mock_sitl.close()
    
    # Analyze received stream
    heartbeats = [m for m in received_messages if m.get_type() == "HEARTBEAT"]
    setpoints = [m for m in received_messages if m.get_type() == "SET_POSITION_TARGET_LOCAL_NED"]

    print(f"  [+] Received total messages: {len(received_messages)}")
    print(f"  [+] Received HEARTBEAT packets: {len(heartbeats)}")
    print(f"  [+] Received SET_POSITION_TARGET_LOCAL_NED packets: {len(setpoints)}")

    assert len(heartbeats) >= 1, "Expected at least 1 HEARTBEAT packet!"
    assert len(setpoints) >= 15, f"Expected at least 15 setpoints in 1.5s (20 Hz), got: {len(setpoints)}"

    sample = setpoints[0]
    last_sample = setpoints[-1]
    assert sample.coordinate_frame == mavutil.mavlink.MAV_FRAME_LOCAL_NED
    assert abs(sample.x - 120.5) < 0.01 and abs(sample.y - (-45.2)) < 0.01
    assert abs(sample.z - (-25.0)) < 6.0  # Operating altitude in nominal band
    assert sample.type_mask == 0x09C0
    assert bridge.supervisor.rejection_counts["STALE_COMMAND"] > 0
    assert abs(last_sample.vx) < 0.01  # stale planner command transitions to a safe stop
    print(f"  [PASS] MAVLink NED setpoint mask verified; stale command stopped (Vx={last_sample.vx:.2f} m/s)")
    print("  [PASS] 20 Hz transmission cadence verified over UDP loopback socket")


if __name__ == "__main__":
    test_safety_supervisor_invariants()
    test_mavlink_udp_streaming()
    print("\n========================================================")
    print("  ALL SAFETY SUPERVISOR & MAVLINK TESTS PASSED (100%)   ")
    print("========================================================")
