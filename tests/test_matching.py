import numpy as np
import pytest

from sage_pose.matching import mask_iou
from sage_pose.matching import contour_chamfer


def test_identical_masks():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert mask_iou(mask, mask) == 1.0


def test_partial_overlap():
    reference = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    candidate = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert mask_iou(reference, candidate) == pytest.approx(1 / 3)


def test_disjoint_masks():
    reference = np.array([[1, 0]], dtype=np.uint8)
    candidate = np.array([[0, 1]], dtype=np.uint8)
    assert mask_iou(reference, candidate) == 0.0


def test_two_empty_masks():
    empty = np.zeros((2, 2), dtype=np.uint8)
    assert mask_iou(empty, empty) == 1.0


def test_rejects_different_sizes():
    with pytest.raises(ValueError, match="mesmo tamanho"):
        mask_iou(np.zeros((2, 2)), np.zeros((3, 3)))


def test_chamfer_identical_masks():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    assert contour_chamfer(mask, mask) == pytest.approx(0.0)


def test_chamfer_increases_with_separation():
    reference = np.zeros((9, 9), dtype=np.uint8)
    near = np.zeros_like(reference)
    far = np.zeros_like(reference)

    reference[4, 2] = 1
    near[4, 3] = 1
    far[4, 6] = 1

    assert contour_chamfer(reference, near) == pytest.approx(1.0)
    assert contour_chamfer(reference, far) == pytest.approx(4.0)


def test_chamfer_empty_masks():
    empty = np.zeros((5, 5), dtype=np.uint8)
    nonempty = empty.copy()
    nonempty[2, 2] = 1

    assert contour_chamfer(empty, empty) == 0.0
    assert np.isinf(contour_chamfer(empty, nonempty))