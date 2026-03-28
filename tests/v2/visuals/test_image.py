"""Tests for ImageAppearance and MultiscaleImageVisual models."""

import uuid

import numpy as np
import pytest
from pydantic import ValidationError

from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._base_visual import BaseAppearance, BaseVisual
from cellier.v2.visuals._image import ImageAppearance, MultiscaleImageVisual


def test_image_appearance_roundtrip(tmp_path):
    # Non-default values including the new LOD/frustum fields
    original = ImageAppearance(
        color_map="viridis",
        clim=(0.1, 0.9),
        lod_bias=2.0,
        force_level=1,
        frustum_cull=False,
    )
    path = tmp_path / "appearance.json"
    path.write_text(original.model_dump_json())
    deserialized = ImageAppearance.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()

    # force_level=None roundtrip
    original_none = ImageAppearance(
        color_map="gray",
        force_level=None,
    )
    path2 = tmp_path / "appearance_none.json"
    path2.write_text(original_none.model_dump_json())
    deserialized_none = ImageAppearance.model_validate_json(path2.read_text())
    assert original_none.model_dump_json() == deserialized_none.model_dump_json()


def test_multiscale_image_visual_roundtrip(tmp_path):
    store_id = str(uuid.uuid4())
    original = MultiscaleImageVisual(
        name="volume",
        data_store_id=store_id,
        downscale_factors=[1, 2, 4],
        appearance=ImageAppearance(
            color_map="viridis",
            clim=(0.0, 1.0),
            lod_bias=1.5,
            force_level=None,
            frustum_cull=True,
        ),
    )
    path = tmp_path / "visual.json"
    path.write_text(original.model_dump_json())
    deserialized = MultiscaleImageVisual.model_validate_json(path.read_text())
    assert original.model_dump_json() == deserialized.model_dump_json()


# ---------------------------------------------------------------------------
# requires_camera_reslice field tests
# ---------------------------------------------------------------------------


class _MinimalVisual(BaseVisual):
    appearance: BaseAppearance = BaseAppearance()


def test_base_visual_requires_camera_reslice_defaults_false():
    v = _MinimalVisual(name="test")
    assert v.requires_camera_reslice is False


def test_multiscale_image_visual_requires_camera_reslice_true():
    v = MultiscaleImageVisual(
        name="vol",
        data_store_id="00000000-0000-0000-0000-000000000000",
        downscale_factors=[1, 2],
        appearance=ImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    assert v.requires_camera_reslice is True


def test_requires_camera_reslice_is_frozen():
    v = MultiscaleImageVisual(
        name="vol",
        data_store_id="00000000-0000-0000-0000-000000000000",
        downscale_factors=[1, 2],
        appearance=ImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    with pytest.raises((ValidationError, TypeError)):
        v.requires_camera_reslice = False


# ---------------------------------------------------------------------------
# Transform field tests
# ---------------------------------------------------------------------------


def test_transform_defaults_to_identity():
    v = _MinimalVisual(name="test")
    np.testing.assert_array_equal(v.transform.matrix, np.eye(4, dtype=np.float32))


def test_transform_field_fires_psygnal():
    v = _MinimalVisual(name="test")
    received = []
    v.events.transform.connect(lambda t: received.append(t))
    new_t = AffineTransform.from_scale((2.0, 2.0, 2.0))
    v.transform = new_t
    assert len(received) == 1
    np.testing.assert_array_equal(received[0].matrix, new_t.matrix)


def test_visual_roundtrip_with_non_identity_transform():
    t = AffineTransform.from_scale_and_translation((2.0, 3.0, 4.0), (10.0, 20.0, 30.0))
    v = MultiscaleImageVisual(
        name="vol",
        data_store_id="00000000-0000-0000-0000-000000000000",
        downscale_factors=[1, 2],
        appearance=ImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
        transform=t,
    )
    json_str = v.model_dump_json()
    v2 = MultiscaleImageVisual.model_validate_json(json_str)
    np.testing.assert_allclose(v2.transform.matrix, t.matrix, atol=1e-6)
