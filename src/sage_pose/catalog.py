"""Carregamento e validação de malhas cadastradas."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yaml


@dataclass(frozen=True)
class ObjectModel:
    object_id: str
    name: str
    units: str
    mesh: trimesh.Trimesh
    dimensions: np.ndarray
    bounds_center: np.ndarray
    is_watertight: bool


def load_object(object_dir: str | Path) -> ObjectModel:
    """Carrega model.ply e metadata.yaml de um objeto cadastrado."""
    object_dir = Path(object_dir)
    model_path = object_dir / "model.ply"
    metadata_path = object_dir / "metadata.yaml"

    if not model_path.is_file():
        raise FileNotFoundError(f"Malha não encontrada: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadados não encontrados: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = yaml.safe_load(file)

    if not isinstance(metadata, dict):
        raise ValueError("metadata.yaml deve conter um mapa de campos.")

    for field in ("object_id", "name", "units"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"Campo obrigatório inválido no metadata.yaml: {field}")

    try:
        mesh = trimesh.load_mesh(model_path, file_type="ply", process=False)
    except Exception as exc:
        raise ValueError(f"Não foi possível ler o PLY: {model_path}") from exc

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("O PLY deve conter uma única malha triangular.")
    if len(mesh.vertices) == 0:
        raise ValueError("O PLY não contém vértices.")
    if len(mesh.faces) == 0:
        raise ValueError("O PLY não contém faces; pode ser apenas uma nuvem de pontos.")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("O PLY contém coordenadas não finitas.")

    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    bounds_center = (bounds[0] + bounds[1]) / 2

    return ObjectModel(
        object_id=metadata["object_id"].strip(),
        name=metadata["name"].strip(),
        units=metadata["units"].strip(),
        mesh=mesh,
        dimensions=dimensions,
        bounds_center=bounds_center,
        is_watertight=bool(mesh.is_watertight),
    )