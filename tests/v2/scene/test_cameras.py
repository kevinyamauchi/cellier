"""Tests for camera and camera controller models."""

import numpy as np

from cellier.scene.cameras import (
    OrbitCameraController,
    OrthographicCamera,
    PanZoomCameraController,
    PerspectiveCamera,
)


def test_orbit_camera_controller_roundtrip(tmp_path):
    original = OrbitCameraController(enabled=False)
    path = tmp_path / "orbit_controller.json"
    path.write_text(original.model_dump_json())
    deserialized = OrbitCameraController.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()


def test_pan_zoom_camera_controller_roundtrip(tmp_path):
    original = PanZoomCameraController(enabled=True)
    path = tmp_path / "panzoom_controller.json"
    path.write_text(original.model_dump_json())
    deserialized = PanZoomCameraController.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()


def test_perspective_camera_roundtrip(tmp_path):
    original = PerspectiveCamera(
        fov=60.0,
        zoom=1.5,
        near_clipping_plane=0.5,
        far_clipping_plane=5000.0,
        position=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        rotation=np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32),
        up_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        frustum=np.ones((2, 4, 3), dtype=np.float32),
        controller=OrbitCameraController(enabled=True),
    )
    path = tmp_path / "perspective_camera.json"
    path.write_text(original.model_dump_json())
    deserialized = PerspectiveCamera.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()


def test_orthographic_camera_roundtrip(tmp_path):
    original = OrthographicCamera(
        width=800.0,
        height=600.0,
        zoom=2.0,
        near_clipping_plane=-1000.0,
        far_clipping_plane=1000.0,
        position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        controller=PanZoomCameraController(enabled=True),
    )
    path = tmp_path / "orthographic_camera.json"
    path.write_text(original.model_dump_json())
    deserialized = OrthographicCamera.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()


def test_frustum_is_derived_and_cannot_be_assigned():
    import pytest

    camera = PerspectiveCamera(controller=OrbitCameraController())
    with pytest.raises(AttributeError, match="frustum"):
        camera.frustum = np.ones((2, 4, 3), dtype=np.float32)
    with camera._mirror_writes():
        camera.frustum = np.ones((2, 4, 3), dtype=np.float32)
    assert camera.frustum.sum() == 24


def test_camera_controller_kind_follows_the_camera():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PerspectiveCamera(controller=PanZoomCameraController())
    with pytest.raises(ValidationError):
        OrthographicCamera(controller=OrbitCameraController())
    camera = PerspectiveCamera(controller=OrbitCameraController())
    with pytest.raises(ValidationError):
        camera.controller.controller_type = "pan_zoom"


def test_camera_assignment_is_validated():
    import pytest
    from pydantic import ValidationError

    camera = OrthographicCamera(controller=PanZoomCameraController())
    with pytest.raises(ValidationError):
        camera.zoom = "far"
    camera.position = [1, 2, 3]
    assert camera.position.dtype == np.float32


def test_a_replaced_controller_still_relays_its_changes():
    camera = PerspectiveCamera(controller=OrbitCameraController())
    seen = []
    camera.events.controller.connect(lambda controller: seen.append(controller.enabled))
    replacement = OrbitCameraController(enabled=True)

    camera.controller = replacement
    replacement.enabled = False

    assert seen == [True, False]


def test_has_pose_is_false_only_for_the_default_pose():
    camera = PerspectiveCamera(controller=OrbitCameraController())
    assert not camera.has_pose
    camera.position = [0.0, 0.0, 5.0]
    assert camera.has_pose
    camera = OrthographicCamera(controller=PanZoomCameraController())
    camera.rotation = [0.0, 0.0, 0.7071068, 0.7071068]
    assert camera.has_pose
