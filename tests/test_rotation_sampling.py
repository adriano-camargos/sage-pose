import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sage_pose.rotation_sampling import euler_rotation_grid


def test_default_grid_size_and_shape():
    rotations = euler_rotation_grid()
    assert rotations.shape == (24 * 13 * 24, 3, 3)


def test_grid_contains_identity():
    rotations = euler_rotation_grid()
    assert np.any(
        np.all(np.isclose(rotations, np.eye(3)), axis=(1, 2))
    )


def test_matrices_are_proper_rotations():
    rotations = euler_rotation_grid(90)
    for matrix in rotations:
        np.testing.assert_allclose(
            matrix.T @ matrix, np.eye(3), atol=1e-12
        )
        assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_contains_ninety_degree_rotation_about_z():
    rotations = euler_rotation_grid(90)
    expected = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    assert np.any(
        np.all(np.isclose(rotations, expected), axis=(1, 2))
    )


@pytest.mark.parametrize("step", [0, -15, 7, 15.0, True])
def test_rejects_invalid_step(step):
    with pytest.raises(ValueError, match="step_degrees"):
        euler_rotation_grid(step)