"""Tests for construction-time visibility on GFX memory/multiscale-label visuals.

Each test verifies that passing ``visible=False`` in the model appearance causes
all nodes built at construction time to start hidden, and ``visible=True``
(the default) produces visible nodes.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from cellier.v2.transform import AffineTransform

# ── GFXImageMemoryVisual ──────────────────────────────────────────────────────


@pytest.fixture
def image_store():
    from cellier.v2.data.image._image_memory_store import ImageMemoryStore

    return ImageMemoryStore(data=np.zeros((4, 4, 4), dtype=np.float32))


def _image_model(store, visible: bool):
    from cellier.v2.visuals._image_memory import ImageVisual, InMemoryImageAppearance

    return ImageVisual(
        name="img",
        data_store_id=str(store.id),
        appearance=InMemoryImageAppearance(color_map="grays", visible=visible),
    )


def test_image_memory_visible_false_2d(image_store):
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    model = _image_model(image_store, visible=False)
    visual = GFXImageMemoryVisual(model, image_store, render_modes={"2d"})

    assert visual.node_2d is not None
    assert visual.node_2d.visible is False


def test_image_memory_visible_false_3d(image_store):
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    model = _image_model(image_store, visible=False)
    visual = GFXImageMemoryVisual(model, image_store, render_modes={"3d"})

    assert visual.node_3d is not None
    assert visual.node_3d.visible is False


def test_image_memory_visible_false_both(image_store):
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    model = _image_model(image_store, visible=False)
    visual = GFXImageMemoryVisual(model, image_store, render_modes={"2d", "3d"})

    assert visual.node_2d.visible is False
    assert visual.node_3d.visible is False


def test_image_memory_visible_true_is_default(image_store):
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    model = _image_model(image_store, visible=True)
    visual = GFXImageMemoryVisual(model, image_store, render_modes={"2d", "3d"})

    assert visual.node_2d.visible is True
    assert visual.node_3d.visible is True


# ── GFXLabelMemoryVisual ──────────────────────────────────────────────────────


@pytest.fixture
def label_store():
    from cellier.v2.data.label._label_memory_store import LabelMemoryStore

    return LabelMemoryStore(data=np.zeros((4, 4, 4), dtype=np.int32))


def _label_memory_model(store, visible: bool):
    from cellier.v2.visuals._label_memory import (
        InMemoryLabelsAppearance,
        LabelMemoryVisual,
    )

    return LabelMemoryVisual(
        name="lbl",
        data_store_id=str(store.id),
        appearance=InMemoryLabelsAppearance(visible=visible),
    )


def test_label_memory_visible_false_2d(label_store):
    from cellier.v2.render.visuals._label_memory import GFXLabelMemoryVisual

    model = _label_memory_model(label_store, visible=False)
    visual = GFXLabelMemoryVisual(model, label_store, render_modes={"2d"})

    assert visual.node_2d is not None
    assert visual.node_2d.visible is False


def test_label_memory_visible_false_3d(label_store):
    from cellier.v2.render.visuals._label_memory import GFXLabelMemoryVisual

    model = _label_memory_model(label_store, visible=False)
    visual = GFXLabelMemoryVisual(model, label_store, render_modes={"3d"})

    assert visual.node_3d is not None
    assert visual.node_3d.visible is False


def test_label_memory_visible_false_both(label_store):
    from cellier.v2.render.visuals._label_memory import GFXLabelMemoryVisual

    model = _label_memory_model(label_store, visible=False)
    visual = GFXLabelMemoryVisual(model, label_store, render_modes={"2d", "3d"})

    assert visual.node_2d.visible is False
    assert visual.node_3d.visible is False


def test_label_memory_visible_true_is_default(label_store):
    from cellier.v2.render.visuals._label_memory import GFXLabelMemoryVisual

    model = _label_memory_model(label_store, visible=True)
    visual = GFXLabelMemoryVisual(model, label_store, render_modes={"2d", "3d"})

    assert visual.node_2d.visible is True
    assert visual.node_3d.visible is True


# ── GFXMultiscaleLabelVisual ──────────────────────────────────────────────────


def _multiscale_label_model(visible: bool):
    from cellier.v2.visuals._labels import (
        MultiscaleLabelsAppearance,
        MultiscaleLabelVisual,
    )

    return MultiscaleLabelVisual(
        name="ms_lbl",
        data_store_id=str(uuid.uuid4()),
        level_transforms=[
            AffineTransform.identity(ndim=3),
            AffineTransform.from_scale_and_translation(
                (2.0, 2.0, 2.0), (0.5, 0.5, 0.5)
            ),
        ],
        appearance=MultiscaleLabelsAppearance(visible=visible),
    )


def test_multiscale_label_visible_false_2d_start():
    from cellier.v2.render.visuals._label_multiscale import GFXMultiscaleLabelVisual

    model = _multiscale_label_model(visible=False)
    level_shapes = [(8, 8, 8), (4, 4, 4)]
    gfx = GFXMultiscaleLabelVisual.from_cellier_model(
        model=model,
        level_shapes=level_shapes,
        render_modes={"2d", "3d"},
        displayed_axes=(1, 2),
    )

    assert gfx.node_2d is not None
    assert gfx.node_2d.visible is False
    assert gfx.node_3d is None


def test_multiscale_label_visible_false_3d_start():
    from cellier.v2.render.visuals._label_multiscale import GFXMultiscaleLabelVisual

    model = _multiscale_label_model(visible=False)
    level_shapes = [(8, 8, 8), (4, 4, 4)]
    gfx = GFXMultiscaleLabelVisual.from_cellier_model(
        model=model,
        level_shapes=level_shapes,
        render_modes={"2d", "3d"},
        displayed_axes=(0, 1, 2),
    )

    assert gfx.node_3d is not None
    assert gfx.node_3d.visible is False
    assert gfx.node_2d is not None  # 2d is also built (axes_2d derived from last 2)
    assert gfx.node_2d.visible is False


def test_multiscale_label_visible_true_is_default():
    from cellier.v2.render.visuals._label_multiscale import GFXMultiscaleLabelVisual

    model = _multiscale_label_model(visible=True)
    level_shapes = [(8, 8, 8), (4, 4, 4)]
    gfx = GFXMultiscaleLabelVisual.from_cellier_model(
        model=model,
        level_shapes=level_shapes,
        render_modes={"2d", "3d"},
        displayed_axes=(0, 1, 2),
    )

    assert gfx.node_3d.visible is True
    assert gfx.node_2d.visible is True
