"""Test the Canvas model."""

import json

import pytest
from pydantic_core import from_json

from cellier.models.scene import (
    Canvas,
    OrbitCameraController,
    OrthographicCamera,
    PerspectiveCamera,
)


@pytest.mark.parametrize("camera_class", [OrthographicCamera, PerspectiveCamera])
def test_canvas(tmp_path, camera_class):
    """Test serialization/deserialiation of the Canvas model."""

    camera = camera_class(controller=OrbitCameraController(enabled=True))
    canvas = Canvas(camera=camera)

    output_path = tmp_path / "test.json"
    with open(output_path, "w") as f:
        # serialize the model
        json.dump(canvas.model_dump(), f)

    # deserialize
    with open(output_path, "rb") as f:
        deserialized_canvas = Canvas.model_validate(
            from_json(f.read(), allow_partial=False)
        )

    assert canvas == deserialized_canvas
