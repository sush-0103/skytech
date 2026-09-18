"""Telemetry-driven geometric route follower for a local SITL experiment.

Positions are map NED metres, times are host monotonic seconds. The independent
gate checks received state and proposed velocity before the transport sends it.
This does not model wind, moving obstacles, jerk or rendered perception.
"""
from dataclasses import dataclass
import math
from src.navigation.osm_world import _distance_to_segment


@dataclass(frozen=True)
class VehicleState:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    received_at: float


class LocalMapFrame:
    def __init__(self, latitude, longitude):
        self.latitude, self.longitude = latitude, longitude
        p = math.radians(latitude)
        e2 = 6.69437999014e-3
        self.north_scale = 6378137*(1-e2)/(1-e2*math.sin(p)**2)**1.5
        self.east_scale = 6378137/math.sqrt(1-e2*math.sin(p)**2)*math.cos(p)

    def to_map(self, latitude, longitude):
        return (math.radians(latitude-self.latitude)*self.north_scale,
                math.radians(longitude-self.longitude)*self.east_scale)

    def to_geographic(self, north, east):
        return (self.latitude+math.degrees(north/self.north_scale),
                self.longitude+math.degrees(east/self.east_scale))


def shortcut_route(world, path, altitude, clearance):
    result, index = [path[0]], 0
    while index < len(path)-1:
        for nxt in range(len(path)-1, index, -1):
            if world.segment_violation((*path[index], -altitude), (*path[nxt], -altitude), clearance) is None:
                result.append(path[nxt])
                index = nxt
                break
        else:
            raise ValueError('No clear continuation')
    return result


class RouteFollower:
    def __init__(self, path, altitude, max_speed=4.0):
        if len(path) < 2 or not all(math.isfinite(v) for p in path for v in p):
            raise ValueError('Need a finite route')
        self.path = path
        self.altitude = altitude
        self.max_speed = max_speed
        self.index = 1
        self.done = False
        self.previous_command = (0., 0., 0.)

    def propose(self, state, dt):
        if not 0 < dt <= .25:
            raise ValueError('Invalid controller interval')
        goal = self.path[self.index]
        delta = (goal[0]-state.position[0], goal[1]-state.position[1])
        distance = math.hypot(*delta)
        speed = math.hypot(*state.velocity[:2])
        if distance < .6 and speed < .35:
            if self.index == len(self.path)-1:
                self.done = True
                return (0., 0., 0.)
            self.index += 1
            return self.propose(state, dt)
        desired_speed = min(self.max_speed, .25*distance)
        wanted = (delta[0]/max(distance, 1e-9)*desired_speed,
                  delta[1]/max(distance, 1e-9)*desired_speed,
                  max(-.8, min(.8, .6*(-self.altitude-state.position[2]))))
        change = tuple(a-b for a,b in zip(wanted, self.previous_command))
        length = math.sqrt(sum(v*v for v in change))
        scale = min(1., 1.0*dt/max(length, 1e-9))  # vector acceleration <= 1 m/s^2
        self.previous_command = tuple(a+scale*b for a,b in zip(self.previous_command,change))
        return self.previous_command

    def cross_track_error(self, position):
        return _distance_to_segment(position[:2], self.path[self.index-1], self.path[self.index])


def gate_velocity(world, state, command, now, route_error, clearance=6.):
    if not all(math.isfinite(v) for v in (*state.position, *state.velocity, state.received_at, *command, now, route_error)):
        return 'NONFINITE_STATE'
    if not 0 <= now-state.received_at <= .3:
        return 'STALE_TELEMETRY'
    if route_error > 2.:
        return 'TRACKING_ERROR'
    if not 4.5 <= -state.position[2] <= 8.:
        return 'ALTITUDE_ERROR'
    if math.hypot(*command[:2]) > 4.01 or abs(command[2]) > .81:
        return 'VELOCITY_LIMIT'
    if math.hypot(*state.velocity[:2]) > 5.5:
        return 'MEASURED_SPEED_LIMIT'
    # Straight stopping corridors for both measured motion and proposed command.
    # 1 m/s² braking and 0.5s reaction allowance are explicit test assumptions.
    for velocity in (state.velocity, command):
        speed = math.sqrt(sum(v*v for v in velocity))
        horizon = .5 + speed/2.
        stop = tuple(p+v*horizon for p,v in zip(state.position, velocity))
        violation = world.segment_violation(state.position, stop, clearance)
        if violation:
            return violation
    return None
