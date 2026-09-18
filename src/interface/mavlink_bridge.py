"""
mavlink_bridge.py
20 Hz MAVLink Setpoint Bridge for ArduPilot SITL.

Per the autonomy architecture in Datasets/autonomous_navigation_system_audit_and_plan.md:
- Operates as a companion computer process.
- Emits accepted 20 Hz MAVLink setpoints (SET_POSITION_TARGET_LOCAL_NED) over local UDP.
- Default target: 127.0.0.1:14550 (configured local ArduPilot SITL endpoint).
- Emits 1 Hz HEARTBEAT (MAV_TYPE_ONBOARD_CONTROLLER).
- Strictly gated by the DeterministicSafetySupervisor.
"""

import time
import math
import socket
import threading
from typing import Dict, Any, Optional, List
from pymavlink import mavutil

from src.interface.safety_supervisor import (
    DeterministicSafetySupervisor,
    SupervisorLimits,
    CandidateSetpoint,
    SupervisorDecision
)


class MAVLinkSITLBridge:
    """
    High-rate 20 Hz MAVLink Companion Computer Bridge for SITL Digital Twins.
    """

    def __init__(
        self,
        target_ip: str = "127.0.0.1",
        target_port: int = 14550,
        rate_hz: float = 20.0,
        system_id: int = 254,          # Companion computer MAVLink sysid
        component_id: int = 191,       # MAV_COMP_ID_ONBOARD_COMPUTER
        supervisor: Optional[DeterministicSafetySupervisor] = None
    ):
        self.target_ip = target_ip
        self.target_port = target_port
        self.rate_hz = rate_hz
        self.period_s = 1.0 / rate_hz
        self.system_id = system_id
        self.component_id = component_id
        
        self.supervisor = supervisor or DeterministicSafetySupervisor()
        
        # Telemetry counters
        self.packets_sent: int = 0
        self.heartbeats_sent: int = 0
        self.setpoints_approved: int = 0
        self.setpoints_rejected: int = 0
        self.last_transmission_time: float = 0.0
        self.actual_rate_hz: float = rate_hz
        self.last_approved_setpoint: Optional[CandidateSetpoint] = None
        self.last_decision: Optional[SupervisorDecision] = None
        self.last_error: Optional[str] = None
        
        # Active candidate setpoint provider
        self.current_candidate: Optional[CandidateSetpoint] = None
        self.current_obstacles: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Connection handle
        self.connection: Optional[Any] = None
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    def initialize_connection(self) -> bool:
        """Create MAVLink UDP connection to SITL."""
        try:
            # Connect via UDP client sending to SITL port
            conn_str = f"udpout:{self.target_ip}:{self.target_port}"
            self.connection = mavutil.mavlink_connection(
                conn_str,
                source_system=self.system_id,
                source_component=self.component_id
            )
            self.last_error = None
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def update_flight_plan(
        self,
        setpoint: CandidateSetpoint,
        obstacles: Optional[List[Dict[str, Any]]] = None
    ):
        """Thread-safe update of current candidate flight setpoint from planner."""
        with self._lock:
            self.current_candidate = setpoint
            if obstacles is not None:
                self.current_obstacles = obstacles

    def start(self):
        """Start the 20 Hz transmission thread."""
        if self._running:
            return
        if not self.initialize_connection():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop transmission thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.connection:
            try:
                self.connection.close()
            except Exception:
                pass

    def _broadcast_loop(self):
        """Main 20 Hz periodic transmission loop."""
        last_heartbeat_time = 0.0
        frame_counter = 0
        t_start_period = time.perf_counter()
        
        # Velocity tracking for acceleration-limited command shaping.
        curr_vx = 0.0
        curr_vy = 0.0
        curr_vz = 0.0
        curr_yaw = 0.0
        last_loop_time = time.time()

        while self._running:
            loop_start = time.perf_counter()
            now = time.time()
            dt = max(0.01, min(0.1, now - last_loop_time))
            last_loop_time = now

            # 1. Heartbeat transmission at 1 Hz
            if now - last_heartbeat_time >= 1.0:
                self._send_heartbeat()
                last_heartbeat_time = now

            # 2. Retrieve latest target setpoint from planner
            with self._lock:
                cand = self.current_candidate
                obs = list(self.current_obstacles)
            # No planner command means heartbeat-only operation. Never invent a position target.
            if cand is None:
                computation_time = time.perf_counter() - loop_start
                time.sleep(max(0.001, self.period_s - computation_time))
                continue
            target_vx = cand.vx
            target_vy = cand.vy
            target_vz = cand.vz
            target_yaw = cand.yaw

            # 3. Kinodynamic acceleration rate-limiter (a_max = 3.5 m/s^2 < 4.0 m/s^2 supervisor limit)
            max_dv = 3.5 * dt
            dv_x = max(-max_dv, min(max_dv, target_vx - curr_vx))
            dv_y = max(-max_dv, min(max_dv, target_vy - curr_vy))
            dv_z = max(-max_dv * 0.6, min(max_dv * 0.6, target_vz - curr_vz))

            curr_vx += dv_x
            curr_vy += dv_y
            curr_vz += dv_z

            curr_yaw = target_yaw

            smooth_candidate = CandidateSetpoint(
                # Preserve producer time so the freshness gate can reject a stale planner.
                timestamp=cand.timestamp,
                x=cand.x,
                y=cand.y,
                z=cand.z,
                vx=curr_vx,
                vy=curr_vy,
                vz=curr_vz,
                yaw=curr_yaw,
                source_model=cand.source_model
            )

            # 4. Independent Deterministic Safety Supervisor Gate
            decision = self.supervisor.evaluate_setpoint(smooth_candidate, current_obstacles=obs, now=now)
            self.last_decision = decision
            approved = decision.approved_setpoint
            self.last_approved_setpoint = approved

            if decision.accepted:
                self.setpoints_approved += 1
            else:
                self.setpoints_rejected += 1
                # On rejection / failsafe, sync internal state with safe setpoint
                curr_vx = approved.vx
                curr_vy = approved.vy
                curr_vz = approved.vz

            # 5. Transmit MAVLink SET_POSITION_TARGET_LOCAL_NED packet (#84)
            self._send_position_target(approved, now)
            self.packets_sent += 1
            self.last_transmission_time = now

            # 5. Rate tracking
            frame_counter += 1
            if frame_counter >= 20:
                elapsed = time.perf_counter() - t_start_period
                if elapsed > 0:
                    self.actual_rate_hz = round(frame_counter / elapsed, 1)
                frame_counter = 0
                t_start_period = time.perf_counter()

            # 6. Sleep to maintain strict 20 Hz cadence (50 ms period)
            computation_time = time.perf_counter() - loop_start
            sleep_duration = max(0.001, self.period_s - computation_time)
            time.sleep(sleep_duration)

    def _send_heartbeat(self):
        """Send MAVLink HEARTBEAT message."""
        if not self.connection:
            return
        try:
            self.connection.mav.heartbeat_send(
                type=mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                autopilot=mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                base_mode=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                custom_mode=0,
                system_status=mavutil.mavlink.MAV_STATE_ACTIVE
            )
            self.heartbeats_sent += 1
        except Exception:
            pass

    def _send_position_target(self, setpoint: CandidateSetpoint, now: float):
        """Send MAVLink SET_POSITION_TARGET_LOCAL_NED message."""
        if not self.connection:
            return
        try:
            time_boot_ms = int((now % 100000) * 1000)
            
            # Control position, velocity and yaw; ignore acceleration and yaw rate.
            # Bits: 0:pos_x, 1:pos_y, 2:pos_z, 3:vx, 4:vy, 5:vz, 6:ax, 7:ay, 8:az, 9:force, 10:yaw, 11:yaw_rate
            type_mask = 0x09C0

            self.connection.mav.set_position_target_local_ned_send(
                time_boot_ms=time_boot_ms,
                target_system=1,         # Primary flight controller sysid
                target_component=1,      # Primary flight controller compid
                coordinate_frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask=type_mask,
                x=float(setpoint.x),
                y=float(setpoint.y),
                z=float(setpoint.z),
                vx=float(setpoint.vx),
                vy=float(setpoint.vy),
                vz=float(setpoint.vz),
                afx=0.0,
                afy=0.0,
                afz=0.0,
                yaw=float(setpoint.yaw),
                yaw_rate=0.0
            )
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive MAVLink SITL telemetry status."""
        supervisor_telemetry = self.supervisor.get_telemetry()
        setpoint_data = {}
        if self.last_approved_setpoint:
            setpoint_data = {
                "x": round(self.last_approved_setpoint.x, 2),
                "y": round(self.last_approved_setpoint.y, 2),
                "z": round(self.last_approved_setpoint.z, 2),
                "vx": round(self.last_approved_setpoint.vx, 2),
                "vy": round(self.last_approved_setpoint.vy, 2),
                "vz": round(self.last_approved_setpoint.vz, 2),
                "yaw_deg": round(math.degrees(self.last_approved_setpoint.yaw), 1),
                "source": self.last_approved_setpoint.source_model
            }

        return {
            "bridge_connected": self._running,
            "connection_error": self.last_error,
            "target_endpoint": f"udp://{self.target_ip}:{self.target_port}",
            "stream_rate_hz": self.actual_rate_hz,
            "target_rate_hz": self.rate_hz,
            "packets_sent": self.packets_sent,
            "heartbeats_sent": self.heartbeats_sent,
            "setpoints_approved": self.setpoints_approved,
            "setpoints_rejected": self.setpoints_rejected,
            "supervisor": supervisor_telemetry,
            "last_approved_setpoint": setpoint_data,
            "last_decision": {
                "accepted": self.last_decision.accepted if self.last_decision else True,
                "state": self.last_decision.state if self.last_decision else "GUIDED_ACTIVE",
                "rejection_reason": self.last_decision.rejection_reason if self.last_decision else None
            } if self.last_decision else None
        }
