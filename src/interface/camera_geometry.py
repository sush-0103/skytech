"""Renderer-independent camera transforms. Metres, radians, NED/FRD.

Camera optical axes: x right, y down, z forward. Gimbal yaw positive
right, pitch positive up. No image rendering or estimated-state substitution.
"""
import math


def multiply(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def rotate(matrix, vector):
    return tuple(sum(row[k] * vector[k] for k in range(3)) for row in matrix)


def attitude(roll, pitch, yaw):
    """Body FRD to NED rotation, ZYX Euler convention."""
    if not all(math.isfinite(v) for v in (roll, pitch, yaw)):
        raise ValueError('Non-finite attitude')
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
            (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr),
            (-sp, cp*sr, cp*cr))


def camera_pose(position_ned, body_rpy, mount_frd=(0.1, 0., 0.05),
                gimbal_pitch=0., gimbal_yaw=0.):
    """Return optical origin and optical-to-NED rotation.

    This ideal kinematic model assumes camera optical centre at gimbal pivot.
    Rate limits, delay and stabilization belong to the simulator controller.
    """
    for vector in (position_ned, body_rpy, mount_frd):
        if len(vector) != 3 or not all(math.isfinite(x) for x in vector):
            raise ValueError('Expected a finite three-vector')
    body = attitude(*body_rpy)
    gimbal = attitude(0., gimbal_pitch, gimbal_yaw)
    optical_to_frd = ((0., 0., 1.), (1., 0., 0.), (0., 1., 0.))
    offset = rotate(body, mount_frd)
    origin = tuple(p + d for p, d in zip(position_ned, offset))
    return origin, multiply(multiply(body, gimbal), optical_to_frd)


def validate_frame_times(rgb_tick, depth_tick, capture_ns, delivery_ns,
                         now_ns, max_age_ns=100_000_000):
    """All times are from the same simulation clock; frozen capture stays old."""
    values = (rgb_tick, depth_tick, capture_ns, delivery_ns, now_ns, max_age_ns)
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError('Expected nonnegative integer ticks/timestamps')
    if rgb_tick != depth_tick:
        raise ValueError('RGB/depth tick mismatch')
    if not capture_ns <= delivery_ns <= now_ns:
        raise ValueError('Invalid clock order')
    if now_ns - capture_ns > max_age_ns:
        raise ValueError('Stale frame')
