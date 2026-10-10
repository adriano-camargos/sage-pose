import numpy as np
import pytest
import trimesh

from sage_pose.catalog import load_object


def write_metadata(object_dir):
    (object_dir / "metadata.yaml").write_text(
        "object_id: test_box\n"
        "name: caixa de teste\n"
        "units: mm\n",
        encoding="utf-8",
    )


def test_load_valid_mesh(tmp_path):
    write_metadata(tmp_path)
    box = trimesh.creation.box(extents=[10.0, 20.0, 30.0])
    box.export(tmp_path / "model.ply")

    model = load_object(tmp_path)

    assert model.object_id == "test_box"
    assert model.name == "caixa de teste"
    assert model.units == "mm"
    assert isinstance(model.mesh, trimesh.Trimesh)
    np.testing.assert_allclose(model.dimensions, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(model.bounds_center, [0.0, 0.0, 0.0])
    assert model.is_watertight


def test_missing_ply(tmp_path):
    write_metadata(tmp_path)

    with pytest.raises(FileNotFoundError, match="Malha não encontrada"):
        load_object(tmp_path)


def test_invalid_ply(tmp_path):
    write_metadata(tmp_path)
    (tmp_path / "model.ply").write_text("isto não é um PLY", encoding="utf-8")

    with pytest.raises(ValueError, match="Não foi possível ler o PLY"):
        load_object(tmp_path)


def test_missing_units(tmp_path):
    box = trimesh.creation.box()
    box.export(tmp_path / "model.ply")
    (tmp_path / "metadata.yaml").write_text(
        "object_id: test_box\nname: caixa de teste\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="units"):
        load_object(tmp_path)