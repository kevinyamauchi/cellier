"""Integration tests for GFXMultiscaleImageVisual paint-texture path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

from cellier.v2.controller import CellierController
from cellier.v2.data.image import OMEZarrImageDataStore
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._image import (
    ImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageVisual,
)


def _make_zarr(path: Path, shape: tuple[int, int] = (64, 64)) -> Path:
    """Create a tiny single-level OME-Zarr v0.5 float32 store."""
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_array("s0", shape=shape, chunks=(32, 32), dtype=np.float32)
    arr[:] = np.zeros(shape, dtype=np.float32)
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "s0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0]}
                        ],
                    }
                ],
                "name": "test",
            }
        ],
    }
    return path


@pytest.fixture
def visual_setup(qtbot, tmp_path):
    """Build a GFXMultiscaleImageVisual backed by a small float32 OME-Zarr."""
    zarr_path = tmp_path / "labels.ome.zarr"
    _make_zarr(zarr_path, shape=(64, 64))
    data_store = OMEZarrImageDataStore.from_path(
        f"file://{zarr_path.resolve()}", name="t"
    )

    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("y", "x"))
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=cs,
        name="paint_scene",
        render_modes={"2d"},
    )

    rc = MultiscaleImageRenderConfig(block_size=16, paint_max_tiles=4)
    visual_model = MultiscaleImageVisual(
        name="t",
        data_store_id=str(data_store.id),
        level_transforms=data_store.level_transforms,
        appearance=ImageAppearance(color_map="grays", clim=(0.0, 1.0)),
        render_config=rc,
        transform=AffineTransform.identity(ndim=2),
    )
    visual = controller.add_visual(scene.id, visual_model, data_store=data_store)
    scene_manager = controller._render_manager._scenes[scene.id]
    gfx_visual = scene_manager.get_visual(visual.id)
    return controller, gfx_visual


def test_paint_textures_are_allocated(visual_setup):
    _ctrl, gfx_visual = visual_setup
    # block_size=16, paint_max_tiles=4 ⇒ paint cache shape (64, 16, 2).
    assert gfx_visual._t_paint_cache.data.shape == (64, 16, 2)
    # LUT shape: 64x64 image ÷ 16-block = 4x4 grid.
    assert gfx_visual._t_paint_lut.data.shape == (4, 4, 2)
    np.testing.assert_array_equal(gfx_visual._t_paint_cache.data, 0.0)
    np.testing.assert_array_equal(gfx_visual._t_paint_lut.data, 0.0)


def test_patch_writes_value_and_alpha(visual_setup):
    _ctrl, gfx_visual = visual_setup
    # Paint two voxels in the same tile (g0=1, g1=2): voxels at (16, 32) and (17, 33).
    voxels = np.array([[16, 32], [17, 33]], dtype=np.int64)
    values = np.array([0.7, 0.4], dtype=np.float32)
    n = gfx_visual.patch_paint_texture(voxels, values, displayed_axes=(0, 1))
    assert n == 1

    # Slot 0 should be allocated to (1, 2).
    assert gfx_visual._paint_slot_manager.get((1, 2)) == 0
    # LUT entry: (slot=0, alpha=1).
    np.testing.assert_array_equal(gfx_visual._t_paint_lut.data[1, 2], [0.0, 1.0])
    # Stripes layout: slot 0 occupies rows 0..16.
    # Voxel (16, 32) → ty=0, tx=0 in tile (1, 2) → cache[0, 0]
    # Voxel (17, 33) → ty=1, tx=1 in tile (1, 2) → cache[1, 1]
    np.testing.assert_allclose(gfx_visual._t_paint_cache.data[0, 0], [0.7, 1.0])
    np.testing.assert_allclose(gfx_visual._t_paint_cache.data[1, 1], [0.4, 1.0])


def test_multiple_tiles_get_distinct_slots(visual_setup):
    _ctrl, gfx_visual = visual_setup
    voxels = np.array([[0, 0], [16, 0], [0, 16]], dtype=np.int64)
    values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    n = gfx_visual.patch_paint_texture(voxels, values, displayed_axes=(0, 1))
    assert n == 3
    sm = gfx_visual._paint_slot_manager
    assert {sm.get((0, 0)), sm.get((1, 0)), sm.get((0, 1))} == {0, 1, 2}


def test_pool_exhaustion_drops_excess_tiles(visual_setup):
    _ctrl, gfx_visual = visual_setup
    # paint_max_tiles=4 — paint into 5 distinct tiles.
    voxels = np.array(
        [[0, 0], [16, 0], [0, 16], [16, 16], [32, 0]],
        dtype=np.int64,
    )
    values = np.array([1.0] * 5, dtype=np.float32)
    n = gfx_visual.patch_paint_texture(voxels, values, displayed_axes=(0, 1))
    assert n == 4
    assert gfx_visual._paint_slot_manager.exhausted is True
    # The 5th tile (2, 0) has no slot.
    assert gfx_visual._paint_slot_manager.get((2, 0)) is None


def test_clear_resets_textures(visual_setup):
    _ctrl, gfx_visual = visual_setup
    voxels = np.array([[0, 0]], dtype=np.int64)
    gfx_visual.patch_paint_texture(
        voxels, np.array([0.5], dtype=np.float32), displayed_axes=(0, 1)
    )
    assert gfx_visual._paint_slot_manager.n_allocated == 1
    gfx_visual.clear_paint_textures()
    np.testing.assert_array_equal(gfx_visual._t_paint_cache.data, 0.0)
    np.testing.assert_array_equal(gfx_visual._t_paint_lut.data, 0.0)
    assert gfx_visual._paint_slot_manager.n_allocated == 0
