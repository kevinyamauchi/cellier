"""Tests for Scene model."""

import uuid

from cellier.v2.scene.cameras import OrbitCameraController, PerspectiveCamera
from cellier.v2.scene.canvas import Canvas
from cellier.v2.scene.dims import CoordinateSystem, DimsManager
from cellier.v2.scene.scene import Scene
from cellier.v2.visuals._image import ImageAppearance, MultiscaleImageVisual


def test_scene_roundtrip(tmp_path):
    dims = DimsManager(
        coordinate_system=CoordinateSystem(name="world", axis_labels=("z", "y", "x")),
        displayed_axes=(0, 1, 2),
        slice_indices=(),
    )
    visual = MultiscaleImageVisual(
        name="volume",
        data_store_id=str(uuid.uuid4()),
        downscale_factors=[1, 2],
        appearance=ImageAppearance(color_map="viridis"),
    )
    camera = PerspectiveCamera(controller=OrbitCameraController())
    canvas = Canvas(cameras={"3d": camera})
    original = Scene(
        name="main",
        dims=dims,
        visuals=[visual],
        canvases={canvas.id: canvas},
    )
    path = tmp_path / "scene.json"
    path.write_text(original.model_dump_json())
    deserialized = Scene.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()
