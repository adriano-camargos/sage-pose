
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sage_pose.yaw_reference import (
    AXIS_ALIGNMENT,
    angular_error_mod180,
    calculate_board_yaw,
    normalize_yaw_180,
)


def rot_z(degrees):
    return Rotation.from_euler(
        "z", degrees, degrees=True
    ).as_matrix()


def test_zero_degrees():
    B = np.eye(3)
    R = B @ AXIS_ALIGNMENT

    result = calculate_board_yaw(R, B)

    assert result["yaw_object_to_board_deg"] == pytest.approx(0)
    assert result["tilt_from_reference_deg"] == pytest.approx(0)


@pytest.mark.parametrize(
    "angle",
    [0, 15, 30, 45, 60, 90, 180, -90],
)
def test_known_angles(angle):
    B = np.eye(3)
    R = B @ AXIS_ALIGNMENT @ rot_z(angle)

    result = calculate_board_yaw(R, B)

    error = (
        (result["yaw_object_to_board_deg"] - angle + 180)
        % 360 - 180
    )

    assert abs(error) < 1e-6


def test_camera_rotation_invariance():
    # A orientacao medida no tabuleiro deve permanecer
    # igual quando ambos os referenciais sofrem a mesma
    # mudanca de orientacao da camera.
    B = Rotation.from_euler(
        "xyz", [20, -35, 70], degrees=True
    ).as_matrix()

    R = B @ AXIS_ALIGNMENT @ rot_z(45)

    result = calculate_board_yaw(R, B)

    assert result["yaw_object_to_board_deg"] == pytest.approx(
        45, abs=1e-6
    )


def test_symmetry_180():
    assert normalize_yaw_180(0) == pytest.approx(0)
    assert normalize_yaw_180(180) == pytest.approx(0)
    assert normalize_yaw_180(-90) == pytest.approx(90)

    assert angular_error_mod180(179.2, 0) == pytest.approx(
        0.8
    )
    assert angular_error_mod180(-88.235, 90) == pytest.approx(
        1.765
    )


def test_invalid_rotation():
    invalid = np.zeros((3, 3))

    with pytest.raises(ValueError):
        calculate_board_yaw(invalid, np.eye(3))
