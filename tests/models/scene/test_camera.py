"""Tests for the camera models."""

import json

from pydantic_core import from_json

from cellier.models.scene import PanZoomCameraController, PerspectiveCamera


def test_perspective_camera(tmp_path):
    """Test the perspective camera serialization/deserialization."""
    camera = PerspectiveCamera(controller=PanZoomCameraController(enabled=True))

    output_path = tmp_path / "test.json"
    with open(output_path, "w") as f:
        # serialize the model
        json.dump(camera.model_dump(), f)

    # deserialize
    with open(output_path, "rb") as f:
        deserialized_camera = PerspectiveCamera.model_validate(
            from_json(f.read(), allow_partial=False)
        )

    assert camera == deserialized_camera
