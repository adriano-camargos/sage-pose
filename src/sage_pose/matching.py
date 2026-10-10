"""Métricas para comparar máscaras de silhueta."""

import numpy as np
import cv2


def mask_iou(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Calcula a interseção sobre união (IoU) de duas máscaras binárias.

    Pixels diferentes de zero são considerados parte do objeto.
    Duas máscaras vazias recebem IoU 1.0.
    """
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)

    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("As máscaras devem ter duas dimensões.")
    if reference.shape != candidate.shape:
        raise ValueError("As máscaras devem ter o mesmo tamanho.")

    reference_fg = reference != 0
    candidate_fg = candidate != 0

    intersection = np.count_nonzero(reference_fg & candidate_fg)
    union = np.count_nonzero(reference_fg | candidate_fg)

    return 1.0 if union == 0 else float(intersection / union)

def contour_chamfer(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Distância média simétrica entre contornos, em pixels.

    Retorna 0.0 se ambas as máscaras estiverem vazias e infinito se
    apenas uma estiver vazia. Menor valor significa melhor correspondência.
    """
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)

    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("As máscaras devem ter duas dimensões.")
    if reference.shape != candidate.shape:
        raise ValueError("As máscaras devem ter o mesmo tamanho.")

    reference_mask = (reference != 0).astype(np.uint8)
    candidate_mask = (candidate != 0).astype(np.uint8)

    if not reference_mask.any() and not candidate_mask.any():
        return 0.0
    if not reference_mask.any() or not candidate_mask.any():
        return float("inf")

    kernel = np.ones((3, 3), dtype=np.uint8)
    reference_edge = reference_mask - cv2.erode(reference_mask, kernel)
    candidate_edge = candidate_mask - cv2.erode(candidate_mask, kernel)

    # distanceTransform calcula a distância até o pixel zero mais próximo.
    distance_to_reference = cv2.distanceTransform(
        1 - reference_edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    distance_to_candidate = cv2.distanceTransform(
        1 - candidate_edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )

    candidate_to_reference = distance_to_reference[candidate_edge != 0].mean()
    reference_to_candidate = distance_to_candidate[reference_edge != 0].mean()

    return float((candidate_to_reference + reference_to_candidate) / 2)