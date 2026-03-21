"""Tests for Canvas model."""

from cellier.v2.scene.cameras import (
    OrbitCameraController,
    OrthographicCamera,
    PanZoomCameraController,
    PerspectiveCamera,
)
from cellier.v2.scene.canvas import Canvas


def test_canvas_single_camera_roundtrip(tmp_path):
    original = Canvas(
        cameras={"3d": PerspectiveCamera(controller=OrbitCameraController())}
    )
    path = tmp_path / "canvas_single.json"
    path.write_text(original.model_dump_json())
    deserialized = Canvas.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()


def test_canvas_two_camera_roundtrip(tmp_path):
    original = Canvas(
        cameras={
            "3d": PerspectiveCamera(controller=OrbitCameraController()),
            "2d": OrthographicCamera(controller=PanZoomCameraController()),
        }
    )
    path = tmp_path / "canvas_two.json"
    path.write_text(original.model_dump_json())
    deserialized = Canvas.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()
