
"""
SAGE-Pose - Yaw relativo ao tabuleiro.

Calcula a orientacao angular do objeto em relacao
a uma referencia fixa construida sobre o tabuleiro.

Convencao:
    R_object_to_camera: objeto -> camera
    R_board_to_camera: tabuleiro -> camera

A orientacao zero experimental foi definida por:
    AXIS_ALIGNMENT = diag(-1, 1, -1)

Essa convencao foi validada inicialmente no conjunto
de seis imagens do balde. Nao e uma convencao universal
para todos os objetos ou tabuleiros.

Nao modifica a matriz de pose 6D estimada.
"""

import numpy as np


AXIS_ALIGNMENT = np.diag([-1.0, 1.0, -1.0])


def validate_rotation_matrix(R, name="R", atol=1e-5):
    """Verifica se R pertence aproximadamente a SO(3)."""
    R = np.asarray(R, dtype=np.float64)

    if R.shape != (3, 3):
        raise ValueError(f"{name} deve ter dimensoes 3x3.")

    if not np.all(np.isfinite(R)):
        raise ValueError(f"{name} contem valores invalidos.")

    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError(f"{name} nao e ortogonal.")

    if not np.isclose(np.linalg.det(R), 1.0, atol=atol):
        raise ValueError(
            f"{name} nao e uma rotacao propria (det != 1)."
        )

    return R


def normalize_yaw_360(angle_deg):
    """Normaliza o angulo para [0, 360)."""
    return float(angle_deg % 360.0)


def normalize_yaw_180(angle_deg):
    """Normaliza o angulo para [0, 180)."""
    return float(angle_deg % 180.0)


def angular_error_mod180(estimated_deg, expected_deg):
    """Retorna erro absoluto considerando simetria de 180 graus."""
    return float(
        abs((estimated_deg - expected_deg + 90.0) % 180.0 - 90.0)
    )


def calculate_board_yaw(
    R_object_to_camera,
    R_board_to_camera,
    axis_alignment=None,
):
    """
    Calcula o yaw relativo a referencia experimental do tabuleiro.

    Returns:
        dict com yaw, matrizes e inclinacao residual.
    """
    R_obj = validate_rotation_matrix(
        R_object_to_camera, "R_object_to_camera"
    )
    R_board = validate_rotation_matrix(
        R_board_to_camera, "R_board_to_camera"
    )

    if axis_alignment is None:
        alignment = AXIS_ALIGNMENT.copy()
    else:
        alignment = validate_rotation_matrix(
            axis_alignment, "axis_alignment"
        )

    # Pose do objeto no referencial do tabuleiro.
    R_object_to_board = R_board.T @ R_obj

    # Orientacao experimental de yaw zero.
    R_zero_to_camera = R_board @ alignment

    # Rotacao relativa entre o zero experimental e a pose.
    R_relative = R_zero_to_camera.T @ R_obj

    # Heading em torno do eixo Z da referencia experimental.
    yaw_rad = np.arctan2(
        R_relative[1, 0],
        R_relative[0, 0],
    )
    yaw_signed_deg = float(
        (np.degrees(yaw_rad) + 180.0) % 360.0 - 180.0
    )

    # Inclinacao do Z local relativa ao Z da referencia.
    z_alignment = float(np.clip(R_relative[2, 2], -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(z_alignment)))

    return {
        "yaw_object_to_board_deg": yaw_signed_deg,
        "yaw_object_to_board_mod180_deg":
            normalize_yaw_180(yaw_signed_deg),
        "R_object_to_board": R_object_to_board.tolist(),
        "R_yaw_zero_to_camera": R_zero_to_camera.tolist(),
        "axis_alignment": alignment.tolist(),
        "tilt_from_reference_deg": tilt_deg,
    }
