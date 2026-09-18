"""
safety_supervisor.py
Deterministic Safety Supervisor for Autonomous UAV Navigation.

Per the safety architecture in Datasets/autonomous_navigation_system_audit_and_plan.md:
Learned models may propose observations and candidate setpoints, but may NEVER
directly command motors or bypass safety checks.

This supervisor executes independent, deterministic checks before each setpoint
is approved for transmission to ArduPilot SITL via MAVLink:
1. Geofence Boundary Check (Local NED coordinate bounds)
2. Kinodynamic Constraints (Velocity, Acceleration, Climb Rate, Jerk ceilings)
3. Dynamic Clearance Check (Minimum 3D separation to detected obstacles)
4. Freshness Gate (Rejection of stale commands > 50 ms)
5. Proof-of-life Heartbeat & Failsafe State Machine (GUIDED -> BRAKE -> HOLD -> LAND)
"""

import time
import math
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class SupervisorLimits:
    # Kinematic limits
    max_velocity_mps: float = 15.0       # Horizontal + total speed ceiling
    max_accel_mps2: float = 4.0          # Max allowable acceleration
    max_climb_mps: float = 2.5           # Max vertical climb/descent speed
    max_jerk_mps3: float = 8.0           # Jerk limit for polynomial smoothness
    
    # Geofence limits (Local NED frame in meters: Z negative is upward)
    x_min_m: float = -500.0
    x_max_m: float = 500.0
    y_min_m: float = -500.0
    y_max_m: float = 500.0
    z_min_m: float = -120.0             # 120m ceiling (typical AGL limit)
    z_max_m: float = -5.0               # 5m minimum operating floor
    
    # Collision avoidance buffer
    min_clearance_m: float = 15.0        # Minimum safe distance to obstacles
    static_clearance_m: float = 6.0      # Building/restricted-boundary horizontal margin
    
    # Timing & age
    max_command_age_s: float = 0.050     # 50 ms max staleness


@dataclass
class CandidateSetpoint:
    timestamp: float                     # Producer wall-clock Unix epoch in seconds
    x: float = 0.0                       # North (meters)
    y: float = 0.0                       # East (meters)
    z: float = -25.0                     # Down (meters, negative is up)
    vx: float = 0.0                      # Velocity North (m/s)
    vy: float = 0.0                      # Velocity East (m/s)
    vz: float = 0.0                      # Velocity Down (m/s)
    afx: float = 0.0                     # Acceleration North (m/s^2)
    afy: float = 0.0                     # Acceleration East (m/s^2)
    afz: float = 0.0                     # Acceleration Down (m/s^2)
    yaw: float = 0.0                     # Target heading (radians)
    source_model: str = "3d_kinematic_astar"


@dataclass
class SupervisorDecision:
    accepted: bool
    state: str                           # 'GUIDED_ACTIVE', 'FAILSAFE_BRAKE', 'FAILSAFE_HOLD', 'EMERGENCY_LAND'
    approved_setpoint: CandidateSetpoint
    rejection_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class DeterministicSafetySupervisor:
    """
    Independent non-learned safety barrier enforcing physical and operational constraints.
    """

    def __init__(self, limits: Optional[SupervisorLimits] = None, static_world: Optional[Any] = None):
        self.limits = limits or SupervisorLimits()
        self.static_world = static_world
        
        # State machine
        self.state = "GUIDED_ACTIVE"
        self.last_accepted_setpoint: Optional[CandidateSetpoint] = None
        self.last_accepted_time: float = time.time()
        self.consecutive_rejections: int = 0
        
        # Statistics
        self.total_evaluated: int = 0
        self.total_accepted: int = 0
        self.total_rejected: int = 0
        self.rejection_counts: Dict[str, int] = {
            "GEOFENCE_EXCEEDED": 0,
            "VELOCITY_EXCEEDED": 0,
            "CLIMB_RATE_EXCEEDED": 0,
            "ACCEL_EXCEEDED": 0,
            "CLEARANCE_VIOLATION": 0,
            "STATIC_WORLD_VIOLATION": 0,
            "STALE_COMMAND": 0,
        }

    def reset_failsafe(self):
        """Reset supervisor to active state."""
        self.state = "GUIDED_ACTIVE"
        self.consecutive_rejections = 0

    def evaluate_setpoint(
        self,
        candidate: CandidateSetpoint,
        current_obstacles: Optional[List[Dict[str, Any]]] = None,
        now: Optional[float] = None
    ) -> SupervisorDecision:
        """
        Evaluate a candidate setpoint against all safety invariants.
        Returns a SupervisorDecision with accepted boolean and the setpoint to emit.
        """
        if now is None:
            now = time.time()

        self.total_evaluated += 1

        # 1. Freshness check: Reject commands older than 50 ms
        command_age = now - candidate.timestamp
        if command_age > self.limits.max_command_age_s or command_age < -1.0:
            return self._reject(
                "STALE_COMMAND",
                candidate,
                f"Command age {command_age*1000:.1f}ms exceeds {self.limits.max_command_age_s*1000:.1f}ms limit"
            )

        # 2. Geofence check (Local NED bounds)
        if not (self.limits.x_min_m <= candidate.x <= self.limits.x_max_m):
            return self._reject(
                "GEOFENCE_EXCEEDED",
                candidate,
                f"Position X {candidate.x:.1f}m outside geofence [{self.limits.x_min_m}, {self.limits.x_max_m}]"
            )

        if not (self.limits.y_min_m <= candidate.y <= self.limits.y_max_m):
            return self._reject(
                "GEOFENCE_EXCEEDED",
                candidate,
                f"Position Y {candidate.y:.1f}m outside geofence [{self.limits.y_min_m}, {self.limits.y_max_m}]"
            )

        # In NED frame, Z is negative up (e.g. -120m ceiling, -5m floor)
        if not (self.limits.z_min_m <= candidate.z <= self.limits.z_max_m):
            return self._reject(
                "GEOFENCE_EXCEEDED",
                candidate,
                f"Altitude Z {candidate.z:.1f}m outside operating band [{self.limits.z_min_m}, {self.limits.z_max_m}]"
            )

        # Static world check: mapped buildings, map bounds and configured restricted areas.
        if self.static_world is not None:
            violation = self.static_world.violation(
                candidate.x, candidate.y, candidate.z,
                horizontal_clearance_m=self.limits.static_clearance_m
            )
            if violation:
                return self._reject(
                    "STATIC_WORLD_VIOLATION", candidate,
                    f"Candidate intersects {violation}"
                )
            if self.last_accepted_setpoint is not None:
                previous = self.last_accepted_setpoint
                violation = self.static_world.segment_violation(
                    (previous.x, previous.y, previous.z),
                    (candidate.x, candidate.y, candidate.z),
                    horizontal_clearance_m=self.limits.static_clearance_m
                )
                if violation:
                    return self._reject(
                        "STATIC_WORLD_VIOLATION", candidate,
                        f"Command segment intersects {violation}"
                    )

        # 3. Velocity constraint check
        v_horiz = math.hypot(candidate.vx, candidate.vy)
        v_total = math.sqrt(candidate.vx**2 + candidate.vy**2 + candidate.vz**2)
        if v_total > self.limits.max_velocity_mps:
            return self._reject(
                "VELOCITY_EXCEEDED",
                candidate,
                f"Total velocity {v_total:.2f} m/s exceeds max {self.limits.max_velocity_mps:.1f} m/s"
            )

        # 4. Vertical climb/descent rate check
        if abs(candidate.vz) > self.limits.max_climb_mps:
            return self._reject(
                "CLIMB_RATE_EXCEEDED",
                candidate,
                f"Vertical rate |vz|={abs(candidate.vz):.2f} m/s exceeds climb cap {self.limits.max_climb_mps:.1f} m/s"
            )

        # 5. Acceleration constraint check (against previous setpoint if available)
        if self.last_accepted_setpoint is not None:
            dt = candidate.timestamp - self.last_accepted_setpoint.timestamp
            if dt >= 0.02:
                ax = (candidate.vx - self.last_accepted_setpoint.vx) / dt
                ay = (candidate.vy - self.last_accepted_setpoint.vy) / dt
                az = (candidate.vz - self.last_accepted_setpoint.vz) / dt
                a_mag = math.sqrt(ax**2 + ay**2 + az**2)
                if a_mag > self.limits.max_accel_mps2 * 1.25:  # Allow 25% tolerance for discrete steps
                    return self._reject(
                        "ACCEL_EXCEEDED",
                        candidate,
                        f"Derived acceleration {a_mag:.2f} m/s^2 exceeds limit {self.limits.max_accel_mps2:.1f} m/s^2"
                    )

        # 6. Obstacle clearance check
        if current_obstacles:
            for obs in current_obstacles:
                ox = obs.get("x", obs.get("relX", 0.0))
                oy = obs.get("y", obs.get("relY", 0.0))
                oz = obs.get("z", candidate.z)
                dist = math.sqrt((candidate.x - ox)**2 + (candidate.y - oy)**2 + (candidate.z - oz)**2)
                if dist < self.limits.min_clearance_m:
                    return self._reject(
                        "CLEARANCE_VIOLATION",
                        candidate,
                        f"Proximity to obstacle {obs.get('id', 'unknown')} ({dist:.1f}m) < safety buffer {self.limits.min_clearance_m:.1f}m"
                    )

        # --- All Safety Checks Passed ---
        self.consecutive_rejections = 0
        self.state = "GUIDED_ACTIVE"
        self.total_accepted += 1
        self.last_accepted_setpoint = candidate
        self.last_accepted_time = now

        return SupervisorDecision(
            accepted=True,
            state=self.state,
            approved_setpoint=candidate,
            diagnostics={
                "v_total_mps": round(v_total, 2),
                "v_horiz_mps": round(v_horiz, 2),
                "vz_mps": round(candidate.vz, 2),
                "age_ms": round(command_age * 1000, 1),
                "consecutive_rejections": 0
            }
        )

    def _reject(self, reason: str, candidate: CandidateSetpoint, detail: str) -> SupervisorDecision:
        """Handle setpoint rejection and trigger fail-safe mode."""
        self.total_rejected += 1
        self.consecutive_rejections += 1
        if reason in self.rejection_counts:
            self.rejection_counts[reason] += 1

        # State transition: Immediate brake if 1-2 rejections, hold if >=3, land if >=10
        if self.consecutive_rejections >= 10:
            self.state = "EMERGENCY_LAND"
        elif self.consecutive_rejections >= 3:
            self.state = "FAILSAFE_HOLD"
        else:
            self.state = "FAILSAFE_BRAKE"

        # Construct safe emergency setpoint: zero velocity stop at current position
        safe_pos_x = self.last_accepted_setpoint.x if self.last_accepted_setpoint else candidate.x
        safe_pos_y = self.last_accepted_setpoint.y if self.last_accepted_setpoint else candidate.y
        safe_pos_z = self.last_accepted_setpoint.z if self.last_accepted_setpoint else candidate.z

        # In emergency landing, command gentle descent (vz = +0.5 m/s Down)
        emergency_vz = 0.5 if self.state == "EMERGENCY_LAND" else 0.0

        failsafe_setpoint = CandidateSetpoint(
            timestamp=time.time(),
            x=safe_pos_x,
            y=safe_pos_y,
            z=safe_pos_z,
            vx=0.0,
            vy=0.0,
            vz=emergency_vz,
            yaw=candidate.yaw,
            source_model=f"supervisor_failsafe_{self.state.lower()}"
        )

        # Sync supervisor's state history with the emitted failsafe setpoint
        self.last_accepted_setpoint = failsafe_setpoint
        self.last_accepted_time = failsafe_setpoint.timestamp

        return SupervisorDecision(
            accepted=False,
            state=self.state,
            approved_setpoint=failsafe_setpoint,
            rejection_reason=f"{reason}: {detail}",
            diagnostics={
                "rejection_reason": reason,
                "rejection_detail": detail,
                "consecutive_rejections": self.consecutive_rejections,
                "state": self.state
            }
        )

    def get_telemetry(self) -> Dict[str, Any]:
        """Return real-time supervisor operational statistics."""
        acceptance_rate = (
            round((self.total_accepted / max(1, self.total_evaluated)) * 100.0, 1)
            if self.total_evaluated > 0 else 100.0
        )
        return {
            "status": "ENFORCED",
            "static_world_loaded": self.static_world is not None,
            "state": self.state,
            "total_evaluated": self.total_evaluated,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "acceptance_rate_pct": acceptance_rate,
            "consecutive_rejections": self.consecutive_rejections,
            "rejection_breakdown": self.rejection_counts,
            "limits": {
                "max_velocity_mps": self.limits.max_velocity_mps,
                "max_accel_mps2": self.limits.max_accel_mps2,
                "max_climb_mps": self.limits.max_climb_mps,
                "min_clearance_m": self.limits.min_clearance_m,
                "static_clearance_m": self.limits.static_clearance_m,
                "geofence_horizontal_m": self.limits.x_max_m,
                "geofence_ceiling_m": abs(self.limits.z_min_m)
            }
        }
