"""Deterministic 2D route planning over the generated Dubai OSM world.

The source geometry is local ENU metres. Public interfaces use MAVLink local NED:
NED north=ENU y, east=ENU x, and altitude AGL=-NED z.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence


Point2 = tuple[float, float]


@dataclass(frozen=True)
class RestrictedZone:
    zone_id: str
    polygon_ned: tuple[Point2, ...]
    reason: str = "configured restricted area"


@dataclass(frozen=True)
class RouteResult:
    path_ned: tuple[Point2, ...]
    altitude_m_agl: float
    length_m: float
    expanded_nodes: int
    resolution_m: float


@dataclass(frozen=True)
class StaticBuilding:
    building_id: str
    height_m: float
    rings_ned: tuple[tuple[Point2, ...], ...]
    bounds: tuple[float, float, float, float]


def _point_in_ring(point: Point2, ring: Sequence[Point2]) -> bool:
    x, y = point
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and x < (xj-xi)*(y-yi)/(yj-yi) + xi:
            inside = not inside
        j = i
    return inside


def _distance_to_segment(point: Point2, a: Point2, b: Point2) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx-ax, by-ay
    denom = dx*dx + dy*dy
    if denom == 0:
        return math.hypot(px-ax, py-ay)
    t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / denom))
    return math.hypot(px-(ax+t*dx), py-(ay+t*dy))


def _distance_to_ring(point: Point2, ring: Sequence[Point2]) -> float:
    return min(_distance_to_segment(point, ring[i], ring[(i+1) % len(ring)])
               for i in range(len(ring)))


def _inside_polygon(point: Point2, rings: Sequence[Sequence[Point2]]) -> bool:
    return bool(rings) and _point_in_ring(point, rings[0]) and not any(
        _point_in_ring(point, hole) for hole in rings[1:])


def _segment_distance(a: Point2, b: Point2, c: Point2, d: Point2) -> float:
    def cross(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    if (min(a[0], b[0]) <= max(c[0], d[0]) and min(c[0], d[0]) <= max(a[0], b[0]) and
        min(a[1], b[1]) <= max(c[1], d[1]) and min(c[1], d[1]) <= max(a[1], b[1]) and
        cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0):
        return 0.0
    return min(_distance_to_segment(a,c,d), _distance_to_segment(b,c,d),
               _distance_to_segment(c,a,b), _distance_to_segment(d,a,b))


class OSMWorld:
    """Static world collision model plus deterministic grid A* routing."""

    def __init__(self, geometry_path: Path | str, manifest_path: Path | str,
                 restricted_zones: Iterable[RestrictedZone] = ()):
        self.geometry_path = Path(geometry_path)
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        raw = json.loads(self.geometry_path.read_text(encoding="utf-8"))
        self.buildings = []
        for feature in raw:
            if feature["kind"] != "building":
                continue
            # Convert every ENU (east, north) ring to NED horizontal (north, east).
            rings = tuple(tuple((float(north), float(east)) for east, north in ring)
                          for ring in feature["rings"])
            outer = rings[0]
            bounds = (min(p[0] for p in outer), min(p[1] for p in outer),
                      max(p[0] for p in outer), max(p[1] for p in outer))
            self.buildings.append(StaticBuilding(feature["id"], float(feature["height_m"]), rings, bounds))
        east0, north0, east1, north1 = self.manifest["local_bounds_m"]
        self.north_bounds = (north0, north1)
        self.east_bounds = (east0, east1)
        self.restricted_zones = tuple(restricted_zones)
        self._violation_cache: dict[tuple[float, ...], Optional[str]] = {}

    @classmethod
    def from_directory(cls, directory: Path | str,
                       restricted_zones: Iterable[RestrictedZone] = ()) -> "OSMWorld":
        directory = Path(directory)
        return cls(directory/"geometry.json", directory/"manifest.json", restricted_zones)

    def violation(self, north: float, east: float, down: float,
                  horizontal_clearance_m: float = 5.0,
                  vertical_clearance_m: float = 3.0) -> Optional[str]:
        if not all(math.isfinite(v) for v in (north, east, down, horizontal_clearance_m, vertical_clearance_m)):
            return 'NONFINITE_POSITION'
        if horizontal_clearance_m < 0 or vertical_clearance_m < 0:
            raise ValueError('Clearances must be nonnegative')
        if len(self._violation_cache) > 50000:
            self._violation_cache.clear()
        key = (north, east, down, horizontal_clearance_m, vertical_clearance_m)
        if key in self._violation_cache:
            return self._violation_cache[key]
        margin = 2.0
        if not (self.north_bounds[0] - margin <= north <= self.north_bounds[1] + margin and
                self.east_bounds[0] - margin <= east <= self.east_bounds[1] + margin):
            result = "MAP_BOUNDS_EXCEEDED"
            self._violation_cache[key] = result
            return result
        point = (north, east)
        for zone in self.restricted_zones:
            if _point_in_ring(point, zone.polygon_ned) or _distance_to_ring(point, zone.polygon_ned) < horizontal_clearance_m:
                result = f"RESTRICTED_ZONE:{zone.zone_id}"
                self._violation_cache[key] = result
                return result
        altitude = -down
        for building in self.buildings:
            if altitude > building.height_m + vertical_clearance_m:
                continue
            n0, e0, n1, e1 = building.bounds
            if not (n0-horizontal_clearance_m <= north <= n1+horizontal_clearance_m and
                    e0-horizontal_clearance_m <= east <= e1+horizontal_clearance_m):
                continue
            rings = building.rings_ned
            inside = _inside_polygon(point, rings)
            near_outer = _distance_to_ring(point, rings[0]) < horizontal_clearance_m
            # A point inside a courtyard remains free unless it is close to a wall.
            inside_hole = any(_point_in_ring(point, hole) for hole in rings[1:])
            near_hole_wall = any(_distance_to_ring(point, hole) < horizontal_clearance_m for hole in rings[1:])
            if inside or near_outer or (inside_hole and near_hole_wall):
                result = f"STATIC_BUILDING:{building.building_id}"
                self._violation_cache[key] = result
                return result
        self._violation_cache[key] = None
        return None

    def is_free(self, point: Point2, altitude_m_agl: float,
                clearance_m: float = 5.0) -> bool:
        return self.violation(point[0], point[1], -altitude_m_agl, clearance_m) is None

    def segment_violation(self, start_ned: tuple[float, float, float],
                          end_ned: tuple[float, float, float],
                          horizontal_clearance_m: float = 5.0,
                          sample_spacing_m: float = 2.0) -> Optional[str]:
        # Analytic horizontal swept-segment checks; spacing retained for API compatibility.
        for point in (start_ned, end_ned):
            violation = self.violation(*point, horizontal_clearance_m=horizontal_clearance_m)
            if violation:
                return violation
        a, b = start_ned[:2], end_ned[:2]
        def near_ring(ring):
            return any(_segment_distance(a,b,ring[i],ring[(i+1)%len(ring)]) <= horizontal_clearance_m
                       for i in range(len(ring)))
        for zone in self.restricted_zones:
            if near_ring(zone.polygon_ned):
                return f'RESTRICTED_ZONE:{zone.zone_id}'
        for building in self.buildings:
            # Conservatively check the entire horizontal segment at its lowest altitude.
            if min(-start_ned[2], -end_ned[2]) > building.height_m + 3.0:
                continue
            n0,e0,n1,e1 = building.bounds
            if (max(a[0],b[0]) < n0-horizontal_clearance_m or min(a[0],b[0]) > n1+horizontal_clearance_m or
                max(a[1],b[1]) < e0-horizontal_clearance_m or min(a[1],b[1]) > e1+horizontal_clearance_m):
                continue
            if any(near_ring(ring) for ring in building.rings_ned):
                return f'STATIC_BUILDING:{building.building_id}'
        return None

    def _segment_is_free(self, a: Point2, b: Point2, altitude_m_agl: float,
                         clearance_m: float, sample_spacing_m: float) -> bool:
        return self.segment_violation((a[0], a[1], -altitude_m_agl),
                                      (b[0], b[1], -altitude_m_agl),
                                      clearance_m, sample_spacing_m) is None

    def plan(self, start_ned: Point2, goal_ned: Point2, altitude_m_agl: float,
             resolution_m: float = 10.0, clearance_m: float = 5.0) -> RouteResult:
        if resolution_m <= 0 or altitude_m_agl <= 0:
            raise ValueError("resolution and altitude must be positive")

        north0, east0 = self.north_bounds[0], self.east_bounds[0]
        def cell(point: Point2) -> tuple[int, int]:
            return (round((point[0]-north0)/resolution_m), round((point[1]-east0)/resolution_m))
        def position(c: tuple[int, int]) -> Point2:
            return (north0+c[0]*resolution_m, east0+c[1]*resolution_m)
        start, goal = cell(start_ned), cell(goal_ned)
        if not self.is_free(start_ned, altitude_m_agl, clearance_m):
            raise ValueError("start is not collision-free")
        if not self.is_free(goal_ned, altitude_m_agl, clearance_m):
            raise ValueError("goal is not collision-free")
        max_i = math.floor((self.north_bounds[1]-north0)/resolution_m)
        max_j = math.floor((self.east_bounds[1]-east0)/resolution_m)
        if not self.is_free(position(start), altitude_m_agl, clearance_m):
            raise ValueError("start grid cell is not collision-free; change resolution or start")
        if not self.is_free(position(goal), altitude_m_agl, clearance_m):
            raise ValueError("goal grid cell is not collision-free; change resolution or goal")

        directions = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
        queue = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {start: 0.0}
        expanded = 0
        while queue:
            _, current = heapq.heappop(queue)
            expanded += 1
            if current == goal:
                break
            for di, dj in directions:
                nxt = (current[0]+di, current[1]+dj)
                if not (0 <= nxt[0] <= max_i and 0 <= nxt[1] <= max_j):
                    continue
                if not self.is_free(position(nxt), altitude_m_agl, clearance_m):
                    continue
                if not self._segment_is_free(position(current), position(nxt), altitude_m_agl,
                                             clearance_m, resolution_m/2):
                    continue
                # Forbid diagonal corner cutting through blocked orthogonal cells.
                if di and dj and (not self.is_free(position((current[0]+di, current[1])), altitude_m_agl, clearance_m) or
                                  not self.is_free(position((current[0], current[1]+dj)), altitude_m_agl, clearance_m)):
                    continue
                new_cost = cost[current] + math.hypot(di, dj)*resolution_m
                if new_cost < cost.get(nxt, math.inf):
                    cost[nxt] = new_cost
                    came_from[nxt] = current
                    h = math.hypot(goal[0]-nxt[0], goal[1]-nxt[1])*resolution_m
                    heapq.heappush(queue, (new_cost+h, nxt))
        else:
            raise RuntimeError("no collision-free route found")

        cells = [goal]
        while cells[-1] != start:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        path = [start_ned, *(position(c) for c in cells[1:-1]), goal_ned]
        # Remove collinear intermediate cells while retaining every turn.
        simplified = [path[0]]
        for i in range(1, len(path)-1):
            a, b, c = simplified[-1], path[i], path[i+1]
            cross = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
            if abs(cross) > 1e-6:
                simplified.append(b)
        simplified.append(path[-1])
        for a, b in zip(simplified, simplified[1:]):
            if not self._segment_is_free(a, b, altitude_m_agl, clearance_m, resolution_m/2):
                raise RuntimeError("route segment failed continuous collision sampling")
        length = sum(math.dist(simplified[i], simplified[i+1]) for i in range(len(simplified)-1))
        return RouteResult(tuple(simplified), altitude_m_agl, length, expanded, resolution_m)
