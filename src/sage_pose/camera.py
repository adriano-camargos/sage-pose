"""Projeção de pontos 3D e retificação de máscaras binárias."""

import cv2
import numpy as np


_VALID_DIST_LENGTHS = {4, 5, 8, 12, 14}


def _validate_camera(
    K: np.ndarray,
    dist: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    K = np.asarray(K, dtype=np.float64)

    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("K deve ser uma matriz 3x3 com valores finitos.")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("As distâncias focais em K devem ser positivas.")
    if not np.allclose(K[2], [0.0, 0.0, 1.0]):
        raise ValueError("A última linha de K deve ser [0, 0, 1].")
    if not np.isclose(K[0, 1], 0.0) or not np.isclose(K[1, 0], 0.0):
        raise ValueError("Esta versão requer termos fora da diagonal iguais a zero em K.")

    if dist is None:
        return K, None

    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    if dist.size == 0:
        return K, None
    if dist.size not in _VALID_DIST_LENGTHS or not np.isfinite(dist).all():
        raise ValueError(
            "dist deve conter 4, 5, 8, 12 ou 14 coeficientes finitos."
        )

    return K, dist.reshape(-1, 1)


def project_points(
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
) -> np.ndarray:
    """Projeta pontos do referencial do objeto em pixels da imagem.

    Convenções:
        p_câmera = R @ p_objeto + t
        x cresce para a direita; y cresce para baixo; z aponta para a cena.
        points_3d e t devem usar a mesma unidade.

    Retorna um array (N, 2), com coordenadas (x, y) em pixels.
    """
    points = np.asarray(points_3d, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    K, dist = _validate_camera(K, dist)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_3d deve ter formato (N, 3).")
    if not np.isfinite(points).all():
        raise ValueError("points_3d contém valores não finitos.")
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError("R deve ser uma matriz 3x3 com valores finitos.")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(R), 1.0, atol=1e-6
    ):
        raise ValueError("R deve ser uma matriz de rotação válida.")
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError("t deve conter três valores finitos.")

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    points_camera = (R @ points.T).T + t
    if np.any(points_camera[:, 2] <= 0):
        raise ValueError("Todos os pontos projetados devem estar à frente da câmera.")

    rvec, _ = cv2.Rodrigues(R)
    pixels, _ = cv2.projectPoints(
        points,
        rvec,
        t.reshape(3, 1),
        K,
        dist,
    )
    return pixels.reshape(-1, 2)


def undistort_mask(
    mask: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None,
) -> np.ndarray:
    """Remove a distorção de uma máscara binária, mantendo seus valores 0/1."""
    K, dist = _validate_camera(K, dist)
    mask = np.asarray(mask)

    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask deve ser uma imagem 2D não vazia.")

    binary = (mask != 0).astype(np.uint8)
    if dist is None or not np.any(dist):
        return binary.copy()

    height, width = binary.shape
    map_x, map_y = cv2.initUndistortRectifyMap(
        K,
        dist,
        None,
        K,
        (width, height),
        cv2.CV_32FC1,
    )
    return cv2.remap(
        binary,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )