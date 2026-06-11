"""The multiscale 2-D proxy image must occupy the same world extent as a
memory ``gfx.Image`` of the same data.

Regression test for the half-pixel offset bug: the tile-proxy node used to be
placed corner-at-origin (spanning world ``[0, W]``) while every other node
(memory image, multiscale volume) is center-at-integer (voxel ``i`` at world
``i``, spanning ``[-0.5, N-0.5]``).  That made overlays misalign by half a
level-0 pixel and made a multiscale image jump when toggled between 2-D and 3-D.
"""

from __future__ import annotations

import uuid

import numpy as np
import pygfx as gfx
import pytest

from cellier.v2.render.visuals._image import GFXMultiscaleImageVisual
from cellier.v2.render.visuals._label_multiscale import GFXMultiscaleLabelVisual
from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._image import MultiscaleImageAppearance, MultiscaleImageVisual
from cellier.v2.visuals._labels import (
    MultiscaleLabelsAppearance,
    MultiscaleLabelVisual,
)


def _make_multiscale_image_node_2d(level_shapes):
    model = MultiscaleImageVisual(
        name="img",
        data_store_id=str(uuid.uuid4()),
        level_transforms=[
            AffineTransform.identity(ndim=3),
            AffineTransform.from_scale_and_translation(
                (2.0, 2.0, 2.0), (0.5, 0.5, 0.5)
            ),
        ],
        appearance=MultiscaleImageAppearance(
            color_map="grays", clim=(0.0, 255.0), visible=True
        ),
    )
    gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
        model=model,
        level_shapes=level_shapes,
        render_modes={"2d", "3d"},
        displayed_axes=(1, 2),  # 2D start -> node_2d built, level-0 is (h, w)
    )
    return gfx_visual.node_2d


def _make_multiscale_label_node_2d(level_shapes):
    model = MultiscaleLabelVisual(
        name="lbl",
        data_store_id=str(uuid.uuid4()),
        level_transforms=[
            AffineTransform.identity(ndim=3),
            AffineTransform.from_scale_and_translation(
                (2.0, 2.0, 2.0), (0.5, 0.5, 0.5)
            ),
        ],
        appearance=MultiscaleLabelsAppearance(visible=True),
    )
    gfx_visual = GFXMultiscaleLabelVisual.from_cellier_model(
        model=model,
        level_shapes=level_shapes,
        render_modes={"2d", "3d"},
        displayed_axes=(1, 2),
    )
    return gfx_visual.node_2d


@pytest.mark.parametrize(
    "make_node_2d",
    [_make_multiscale_image_node_2d, _make_multiscale_label_node_2d],
    ids=["image", "labels"],
)
def test_multiscale_2d_node_matches_memory_image_world_extent(make_node_2d) -> None:
    # Non-square, anisotropic-shape level-0 so a W/H or axis-order mistake shows.
    d, h, w = 4, 8, 16
    node_2d = make_node_2d([(d, h, w), (d // 2, h // 2, w // 2)])
    assert node_2d is not None
    ms_bbox = node_2d.get_world_bounding_box()

    # A plain memory image over real (H, W) data — pygfx places pixel i center
    # at i, so it spans [-0.5, W-0.5] x [-0.5, H-0.5].
    mem_image = gfx.Image(
        gfx.Geometry(grid=gfx.Texture(np.zeros((h, w), dtype=np.float32), dim=2)),
        gfx.ImageBasicMaterial(clim=(0, 255)),
    )
    mem_bbox = mem_image.get_world_bounding_box()

    # Compare the displayed (x, y) extent; ignore the degenerate z slab.
    np.testing.assert_allclose(ms_bbox[:, :2], mem_bbox[:, :2], atol=1e-5)

    # And it is the explicit center-at-integer extent [-0.5, N-0.5].
    np.testing.assert_allclose(ms_bbox[0, :2], [-0.5, -0.5], atol=1e-5)
    np.testing.assert_allclose(ms_bbox[1, :2], [w - 0.5, h - 0.5], atol=1e-5)
