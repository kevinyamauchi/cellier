"""Tests for cellier.convenience.Viewer."""

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.render._config import RenderManagerConfig, SlicingConfig
from cellier.visuals._image_memory import InMemoryImageAppearance


@pytest.fixture
def image_store() -> ImageMemoryStore:
    data = np.zeros((8, 16, 16), dtype=np.float32)
    return ImageMemoryStore(data=data, name="test_image")


def test_serialization_roundtrip(tmp_path, image_store):
    """Viewer serializes and deserializes with equivalent ViewerModels."""
    viewer = Viewer(
        axis_labels=("z", "y", "x"),
        dim="2d",
        render_config=RenderManagerConfig(slicing=SlicingConfig(batch_size=16)),
    )
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
        name="test_image",
    )

    path = tmp_path / "viewer.json"
    viewer.to_file(path)

    loaded = Viewer.from_file(path)

    assert viewer.controller._model == loaded.controller._model
