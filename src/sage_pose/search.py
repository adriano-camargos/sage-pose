"""Busca inicial de orientação por comparação de silhuetas."""

from dataclasses import dataclass

import numpy as np
import trimesh

from sage_pose.matching import contour_chamfer, mask_iou
from sage_pose.renderer import render_silhouette


@dataclass(frozen=True)
class RotationMatch:
    """Resultado de uma rotação candidata."""

    index: int
    rotation: np.ndarray
    iou: float
    chamfer_px: float


def search_rotations(
    mesh: trimesh.Trimesh,
    observed_mask: np.ndarray,
    rotations: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    *,
    dist: np.ndarray | None = None,
    top_k: int = 5,
) -> list[RotationMatch]:
    """Classifica rotações com translação conhecida.

    `rotations` tem formato (N, 3, 3). A máscara observada define o
    tamanho da imagem. Retorna primeiro o maior IoU e, em caso de empate,
    o menor Chamfer. Candidatos cuja silhueta fica vazia são descartados.
    """
    observed_mask = np.asarray(observed_mask)
    rotations = np.asarray(rotations)

    if observed_mask.ndim != 2 or not np.any(observed_mask):
        raise ValueError("observed_mask deve ser uma máscara 2D não vazia.")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations deve ter formato (N, 3, 3).")
    if len(rotations) == 0:
        raise ValueError("rotations não pode estar vazio.")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k deve ser um inteiro positivo.")

    image_size = observed_mask.shape
    results: list[RotationMatch] = []

    for index, rotation in enumerate(rotations):
        candidate_mask = render_silhouette(
            mesh=mesh,
            R=rotation,
            t=t,
            K=K,
            image_size=image_size,
            dist=dist,
        )
        if not np.any(candidate_mask):
            continue

        results.append(
            RotationMatch(
                index=index,
                rotation=rotation.copy(),
                iou=mask_iou(observed_mask, candidate_mask),
                chamfer_px=contour_chamfer(observed_mask, candidate_mask),
            )
        )

    results.sort(key=lambda result: (-result.iou, result.chamfer_px, result.index))
    return results[:top_k]