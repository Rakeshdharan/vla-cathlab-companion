import numpy as np
import pytest
from dataengine.geometry import (quat_from_axis_angle, quat_mul, quat_rotate, quat_to_axis_angle, quat_yaw,
                                 rot_error, yaw_quat)


def test_axis_angle_roundtrip():
    v = np.array([0.1, -0.4, 0.3])
    np.testing.assert_allclose(quat_to_axis_angle(quat_from_axis_angle(v)), v, atol=1e-9)


def test_rotate_x_by_90_maps_z_to_minus_y():
    q = quat_from_axis_angle(np.array([np.pi / 2, 0, 0]))
    np.testing.assert_allclose(quat_rotate(q, np.array([0, 0, 1.0])), [0, -1, 0], atol=1e-9)


def test_yaw_roundtrip():
    assert quat_yaw(yaw_quat(0.7)) == pytest.approx(0.7)


def test_rot_error_takes_current_to_target():
    cur = yaw_quat(0.2)
    tgt = quat_mul(quat_from_axis_angle(np.array([0, 0.3, 0])), cur)
    err = rot_error(tgt, cur)
    np.testing.assert_allclose(quat_mul(quat_from_axis_angle(err), cur), tgt, atol=1e-9)
