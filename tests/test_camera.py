import cv2
import numpy as np
import pytest

from sage_pose.camera import project_points, undistort_mask


@pytest.fixture
def K():
    return np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_projection_without_distortion(K):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ]
    )
    R = np.eye(3)
    t = np.array([0.0, 0.0, 1000.0])

    pixels = project_points(points, R, t, K)

    expected = np.array(
        [
            [320.0, 240.0],
            [328.0, 240.0],
            [320.0, 248.0],
        ]
    )
    np.testing.assert_allclose(pixels, expected, atol=1e-10)


def test_projection_with_rotation_and_distortion_matches_opencv(K):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [20.0, 10.0, 5.0],
            [-15.0, 30.0, -5.0],
        ]
    )
    rvec = np.array([0.1, -0.2, 0.3])
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([10.0, -20.0, 500.0])
    dist = np.array([0.1, -0.05, 0.001, -0.002, 0.01])

    pixels = project_points(points, R, t, K, dist)
    expected, _ = cv2.projectPoints(
        points,
        rvec,
        t.reshape(3, 1),
        K,
        dist,
    )

    assert pixels.shape == (3, 2)
    np.testing.assert_allclose(pixels, expected.reshape(-1, 2), atol=1e-8)


def test_rejects_point_behind_camera(K):
    points = np.array([[0.0, 0.0, -2.0]])

    with pytest.raises(ValueError, match="à frente da câmera"):
        project_points(points, np.eye(3), np.zeros(3), K)


def test_rejects_invalid_rotation(K):
    invalid_R = np.diag([2.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="rotação válida"):
        project_points(
            np.array([[0.0, 0.0, 0.0]]),
            invalid_R,
            np.array([0.0, 0.0, 1000.0]),
            K,
        )


def test_undistort_mask_without_distortion(K):
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 10:20] = 255

    result = undistort_mask(mask, K, None)

    assert result.shape == mask.shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)) == {0, 1}
    np.testing.assert_array_equal(result, mask != 0)


def test_undistort_mask_with_distortion_stays_binary(K):
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[180:300, 260:380] = 1
    dist = np.array([0.1, -0.05, 0.001, -0.002, 0.01])

    result = undistort_mask(mask, K, dist)

    assert result.shape == mask.shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 1})
    assert result.any()