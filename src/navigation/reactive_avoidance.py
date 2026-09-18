"""
src/navigation/reactive_avoidance.py
Deterministic 3-Way Evasion Manifold Evaluator for Urban UAV Operations.
Evaluates Left, Right, and Top bypass trajectories around detected obstacles/buildings.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import math


@dataclass(frozen=True)
class CandidateEscapePath:
    direction: str          # "LEFT", "RIGHT", "TOP"
    clearance_m: float     # Estimated minimum distance to obstacle boundary
    cost: float            # Kinodynamic maneuver cost (lower is better)
    primitive: str         # Kinematic primitive name
    lateral_offset_m: float
    vertical_offset_m: float
    status: str            # "OPTIMAL", "FEASIBLE", "CONSTRAINED", "BLOCKED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReactiveAvoidanceDecision:
    obstacle_detected: bool
    obstacle_type: str
    obstacle_id: str
    distance_m: float
    ttc_s: float
    threat_level: str       # "CLEAR", "CAUTION", "CRITICAL"
    bbox: Dict[str, float]  # top, left, width, height (percentage 0-100)
    candidate_paths: List[CandidateEscapePath]
    optimal_path: str       # "LEFT", "RIGHT", "TOP"
    optimal_action: str
    optimal_primitive: str
    target_velocity: Tuple[float, float, float]
    clearance_margin_m: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obstacle_detected": self.obstacle_detected,
            "obstacle_type": self.obstacle_type,
            "obstacle_id": self.obstacle_id,
            "distance_m": round(self.distance_m, 1),
            "ttc_s": round(self.ttc_s, 1),
            "threat_level": self.threat_level,
            "bbox": self.bbox,
            "candidate_paths": [p.to_dict() for p in self.candidate_paths],
            "optimal_path": self.optimal_path,
            "optimal_action": self.optimal_action,
            "optimal_primitive": self.optimal_primitive,
            "target_velocity": [round(v, 2) for v in self.target_velocity],
            "clearance_margin_m": round(self.clearance_margin_m, 1)
        }


def evaluate_escape_manifolds(
    obstacle_distance_m: float,
    obstacle_rel_y: float,            # Lateral offset: <0 is Left, >0 is Right
    obstacle_height_m: float,
    uav_speed_mps: float = 4.0,
    uav_alt_agl_m: float = 6.0,
    min_clearance_m: float = 5.0,
    left_blocked: bool = False,
    right_blocked: bool = False
) -> ReactiveAvoidanceDecision:
    """
    Evaluates Left, Right, and Top bypass trajectories given obstacle geometry.
    Returns ReactiveAvoidanceDecision with the optimal evasion manifold selected.
    """
    uav_speed = max(0.5, uav_speed_mps)
    ttc = obstacle_distance_m / uav_speed

    # Threat categorization
    if obstacle_distance_m <= 15.0 or ttc <= 3.8:
        threat_level = "CRITICAL"
    elif obstacle_distance_m <= 30.0 or ttc <= 7.5:
        threat_level = "CAUTION"
    else:
        threat_level = "CLEAR"

    # Candidate 1: Veer Left (Port evasion)
    # If obstacle is to the right (rel_y > 0), clearing left is spacious
    left_clearance = 15.0 - max(0.0, -obstacle_rel_y) if not left_blocked else 2.5
    left_clearance = max(1.0, left_clearance)
    if left_blocked or left_clearance < min_clearance_m:
        left_cost = 99.0
        left_status = "BLOCKED"
    else:
        # Cost is higher if obstacle is already on the left
        steering_penalty = 12.0 if obstacle_rel_y < 0 else 2.0
        left_cost = (10.0 / left_clearance) + steering_penalty
        left_status = "FEASIBLE"

    left_path = CandidateEscapePath(
        direction="LEFT",
        clearance_m=round(left_clearance, 1),
        cost=round(left_cost, 2),
        primitive="LATERAL_EVADE_WEST",
        lateral_offset_m=-8.5,
        vertical_offset_m=0.0,
        status=left_status
    )

    # Candidate 2: Veer Right (Starboard evasion)
    # If obstacle is to the left (rel_y < 0), clearing right is spacious
    right_clearance = 15.0 - max(0.0, obstacle_rel_y) if not right_blocked else 2.5
    right_clearance = max(1.0, right_clearance)
    if right_blocked or right_clearance < min_clearance_m:
        right_cost = 99.0
        right_status = "BLOCKED"
    else:
        steering_penalty = 12.0 if obstacle_rel_y > 0 else 2.0
        right_cost = (10.0 / right_clearance) + steering_penalty
        right_status = "FEASIBLE"

    right_path = CandidateEscapePath(
        direction="RIGHT",
        clearance_m=round(right_clearance, 1),
        cost=round(right_cost, 2),
        primitive="BANK_RIGHT_EVADE",
        lateral_offset_m=8.5,
        vertical_offset_m=0.0,
        status=right_status
    )

    # Candidate 3: Climb Top (Altitude separation over building roof)
    alt_gap_needed = max(0.0, (obstacle_height_m + min_clearance_m) - uav_alt_agl_m)
    # In ASCEND_RAPID_CLEAR, drone pitches up and decelerates forward (speed * 0.7)
    effective_ttc = obstacle_distance_m / max(0.5, uav_speed * 0.7)
    max_climb_rate = 1.4  # m/s aggressive vertical climb
    climb_time = alt_gap_needed / max_climb_rate
    if climb_time > effective_ttc:
        # Cannot climb fast enough before impact
        top_clearance = 2.0
        top_cost = 85.0
        top_status = "CONSTRAINED"
    else:
        top_clearance = min_clearance_m + 2.5
        # Cost depends on climb height: low buildings are cheap to hop over
        top_cost = 4.0 + (alt_gap_needed * 0.6)
        top_status = "FEASIBLE"


    top_path = CandidateEscapePath(
        direction="TOP",
        clearance_m=round(top_clearance, 1),
        cost=round(top_cost, 2),
        primitive="ASCEND_RAPID_CLEAR",
        lateral_offset_m=0.0,
        vertical_offset_m=round(alt_gap_needed, 1),
        status=top_status
    )

    candidates = [left_path, right_path, top_path]

    # Select optimal path (lowest cost among unblocked/feasible paths)
    valid_candidates = [c for c in candidates if c.status != "BLOCKED"]
    if not valid_candidates:
        optimal = min(candidates, key=lambda c: c.cost)
    else:
        optimal = min(valid_candidates, key=lambda c: c.cost)

    # Mark optimal in candidates
    updated_candidates = []
    for c in candidates:
        if c.direction == optimal.direction:
            updated_candidates.append(CandidateEscapePath(
                direction=c.direction,
                clearance_m=c.clearance_m,
                cost=c.cost,
                primitive=c.primitive,
                lateral_offset_m=c.lateral_offset_m,
                vertical_offset_m=c.vertical_offset_m,
                status="OPTIMAL"
            ))
        else:
            updated_candidates.append(c)

    # Compute target velocity
    if optimal.direction == "RIGHT":
        action_desc = "VEER RIGHT (+18°)"
        t_vx, t_vy, t_vz = (uav_speed * 0.85, 2.2, 0.0)
    elif optimal.direction == "LEFT":
        action_desc = "VEER LEFT (-22°)"
        t_vx, t_vy, t_vz = (uav_speed * 0.85, -2.2, 0.0)
    else:
        action_desc = f"CLIMB TOP (+{round(optimal.vertical_offset_m, 1)}m)"
        t_vx, t_vy, t_vz = (uav_speed * 0.7, 0.0, -0.8)

    # Estimate normalized 2D bounding box on camera screen (0-100%)
    proximity_scale = max(0.2, min(1.0, 35.0 / max(5.0, obstacle_distance_m)))
    box_w = round(22.0 * proximity_scale + 12.0, 1)
    box_h = round(38.0 * proximity_scale + 18.0, 1)

    # Horizontal position in camera view based on obstacle_rel_y
    center_x = 50.0 + (obstacle_rel_y / 15.0) * 32.0
    center_x = max(10.0, min(80.0, center_x))
    box_left = round(center_x - box_w / 2.0, 1)
    box_top = round(max(8.0, 48.0 - box_h / 2.0), 1)

    return ReactiveAvoidanceDecision(
        obstacle_detected=True,
        obstacle_type="Building",
        obstacle_id="BLDG-158-OSM",
        distance_m=obstacle_distance_m,
        ttc_s=ttc,
        threat_level=threat_level,
        bbox={
            "top": box_top,
            "left": box_left,
            "width": box_w,
            "height": box_h
        },
        candidate_paths=updated_candidates,
        optimal_path=optimal.direction,
        optimal_action=action_desc,
        optimal_primitive=optimal.primitive,
        target_velocity=(t_vx, t_vy, t_vz),
        clearance_margin_m=optimal.clearance_m
    )
