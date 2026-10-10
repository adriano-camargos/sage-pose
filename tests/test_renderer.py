import numpy as np
import pytest
import trimesh

from sage_pose.renderer import render_silhouette


@pytest.fixture
def camera():
    K = np.array(
        [
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 50.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return K


@pytest.fixture
def box():
    return trimesh.creation.box(extents=[2.0, 2.0, 2.0])


def test_box_silhouette_has_expected_region(box, camera):
    mask = render_silhouette(
        mesh=box,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 5.0]),
        K=camera,
        image_size=(100, 100),
    )

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 1}

    # A face mais próxima está em z=4. Seus limites projetados
    # são x,y = 50 +/- (100 / 4), ou seja, 25 a 75 pixels.
    assert mask[50, 50] == 1
    assert mask[25, 25] == 1
    assert mask[75, 75] == 1
    assert mask[0, 0] == 0
    assert mask[99, 99] == 0


def test_object_outside_image_returns_empty_mask(box, camera):
    mask = render_silhouette(
        mesh=box,
        R=np.eye(3),
        t=np.array([100.0, 0.0, 5.0]),
        K=camera,
        image_size=(100, 100),
    )

    assert np.count_nonzero(mask) == 0


def test_rejects_mesh_behind_camera(box, camera):
    with pytest.raises(ValueError, match="à frente da câmera"):
        render_silhouette(
            mesh=box,
            R=np.eye(3),
            t=np.array([0.0, 0.0, -5.0]),
            K=camera,
            image_size=(100, 100),
        )


def test_rejects_invalid_image_size(box, camera):
    with pytest.raises(ValueError, match="image_size"):
        render_silhouette(
            mesh=box,
            R=np.eye(3),
            t=np.array([0.0, 0.0, 5.0]),
            K=camera,
            image_size=(0, 100),
        )


def test_render_with_distortion_stays_binary(box, camera):
    dist = np.array([0.1, -0.05, 0.001, -0.002, 0.01])

    mask = render_silhouette(
        mesh=box,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 5.0]),
        K=camera,
        image_size=(100, 100),
        dist=dist,
    )

    assert mask.shape == (100, 100)
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask[50, 50] == 1