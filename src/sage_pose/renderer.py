"""Renderização de silhuetas de malhas triangulares em CPU."""

import cv2
import numpy as np
import trimesh

from sage_pose.camera import project_points


def render_silhouette(
    mesh: trimesh.Trimesh,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    dist: np.ndarray | None = None,
) -> np.ndarray:
    """Projeta as faces da malha e retorna sua silhueta binária.

    Args:
        mesh: Malha triangular no referencial do objeto.
        R, t: Transformação objeto → câmera: p_câmera = R @ p_objeto + t.
        K: Matriz intrínseca 3x3.
        image_size: (altura, largura), em pixels.
        dist: Coeficientes de distorção da câmera ou None.

    A malha inteira deve estar à frente da câmera. As unidades de t devem
    ser iguais às unidades das coordenadas da malha.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh deve ser uma instância de trimesh.Trimesh.")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("A malha deve conter vértices e faces triangulares.")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("A malha contém coordenadas não finitas.")

    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(type(value) is not int or value <= 0 for value in image_size)
    ):
        raise ValueError("image_size deve ser (altura, largura) com inteiros positivos.")

    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    # project_points valida R, t, K, dist e se todos os vértices estão
    # à frente da câmera. Projetar uma vez evita repetir o cálculo por face.
    pixels = project_points(mesh.vertices, R, t, K, dist)

    for face in mesh.faces:
        triangle = pixels[face]

        # Faces totalmente fora da imagem não precisam ser desenhadas.
        if (
            np.max(triangle[:, 0]) < 0
            or np.min(triangle[:, 0]) >= width
            or np.max(triangle[:, 1]) < 0
            or np.min(triangle[:, 1]) >= height
        ):
            continue

        vertices_px = np.rint(triangle).astype(np.int32)
        cv2.fillConvexPoly(mask, vertices_px, color=1)

    return mask