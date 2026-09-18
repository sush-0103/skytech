import math
import unittest
from src.interface.camera_geometry import camera_pose, rotate, validate_frame_times


class CameraGeometryTests(unittest.TestCase):
    def assertVector(self, actual, expected):
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=9)

    def test_forward_and_mount(self):
        p, r = camera_pose((10., 20., -5.), (0., 0., 0.))
        self.assertVector(p, (10.1, 20., -4.95))
        self.assertVector(rotate(r, (0., 0., 1.)), (1., 0., 0.))
        self.assertVector(rotate(r, (1., 0., 0.)), (0., 1., 0.))

    def test_body_turn_rotates_mount_and_view(self):
        p, r = camera_pose((0., 0., 0.), (0., 0., math.pi/2))
        self.assertVector(p, (0., 0.1, 0.05))
        self.assertVector(rotate(r, (0., 0., 1.)), (0., 1., 0.))

    def test_gimbal_up_down_left_right(self):
        for pitch, yaw, expected in (
            (math.pi/2, 0., (0., 0., -1.)),
            (-math.pi/2, 0., (0., 0., 1.)),
            (0., math.pi/2, (0., 1., 0.)),
            (0., -math.pi/2, (0., -1., 0.))):
            p, r = camera_pose((0., 0., 0.), (0., 0., 0.),
                               gimbal_pitch=pitch, gimbal_yaw=yaw)
            self.assertVector(p, (0.1, 0., 0.05))
            self.assertVector(rotate(r, (0., 0., 1.)), expected)

    def test_climb_moves_camera(self):
        p, _ = camera_pose((0., 0., -10.), (0., 0., 0.))
        self.assertAlmostEqual(p[2], -9.95)

    def test_fresh_bundle(self):
        validate_frame_times(4, 4, 0, 20_000_000, 50_000_000)

    def test_invalid_bundles(self):
        for args in ((4, 5, 0, 1, 2), (4, 4, 5, 1, 6),
                     (4, 4, 0, 1, 100_000_001), (4, 4, -1, 1, 2)):
            with self.assertRaises(ValueError):
                validate_frame_times(*args)

    def test_nonfinite_pose_rejected(self):
        with self.assertRaises(ValueError):
            camera_pose((math.nan, 0., 0.), (0., 0., 0.))


if __name__ == '__main__':
    unittest.main()
