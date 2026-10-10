"""Amostragem inicial de rotações para o baseline SAGE-Pose."""

from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation


def euler_rotation_grid(step_degrees: int = 15) -> np.ndarray:
    """Gera matrizes de rotação numa grade de ângulos Euler.

    Convenção: ângulos (roll, pitch, yaw), em graus, com rotação
    extrínseca xyz. Assim, R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    A transformação objeto → câmera continua sendo:
        p_camera = R @ p_objeto + t

    Retorna um array de formato (N, 3, 3). Esta grade é um baseline:
    não distribui orientações uniformemente em SO(3).
    """
    if (
        type(step_degrees) is not int
        or step_degrees <= 0
        or 180 % step_degrees != 0
    ):
        raise ValueError(
            "step_degrees deve ser um inteiro positivo que divide 180."
        )

    roll = range(-180, 180, step_degrees)
    pitch = range(-90, 91, step_degrees)
    yaw = range(-180, 180, step_degrees)

    angles = np.asarray(
        list(product(roll, pitch, yaw)), dtype=np.float64
    )
    return Rotation.from_euler("xyz", angles, degrees=True).as_matrix()