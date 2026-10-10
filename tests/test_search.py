import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from sage_pose.renderer import render_silhouette
from sage_pose.search import search_rotations


@pytest.fixture
def scene():
    # Caixa não quadrada: sua silhueta muda ao girar 90° em torno de Z.
    mesh = trimesh.creation.box(extents=[2.0, 1.0, 1.0])
    K = np.array([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0],
    ])
    t = np.array([0.0, 0.0, 5.0])
    identity = np.eye(3)
    quarter_turn = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    observed = render_silhouette(mesh, identity, t, K, (100, 100))
    return mesh, K, t, identity, quarter_turn, observed


def test_finds_known_rotation(scene):
    mesh, K, t, identity, quarter_turn, observed = scene
    rotations = np.stack([quarter_turn, identity])

    results = search_rotations(
        mesh, observed, rotations, t, K, top_k=2
    )

    assert len(results) == 2
    assert results[0].index == 1
    assert results[0].iou == pytest.approx(1.0)
    assert results[0].chamfer_px == pytest.approx(0.0)
    assert results[1].iou < results[0].iou


def test_top_k_limits_results(scene):
    mesh, K, t, identity, quarter_turn, observed = scene
    rotations = np.stack([quarter_turn, identity])

    results = search_rotations(
        mesh, observed, rotations, t, K, top_k=1
    )

    assert len(results) == 1
    assert results[0].index == 1


def test_rejects_empty_observation(scene):
    mesh, K, t, identity, _, observed = scene

    with pytest.raises(ValueError, match="observed_mask"):
        search_rotations(
            mesh, np.zeros_like(observed), identity[None], t, K
        )


def test_rejects_invalid_rotation_shape(scene):
    mesh, K, t, _, _, observed = scene

    with pytest.raises(ValueError, match="rotations"):
        search_rotations(mesh, observed, np.zeros((3, 3)), t, K)