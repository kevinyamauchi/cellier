"""Planning from the region produces what planning from ``dims_state`` did.

Design 3.7's claim, checked on every in-memory family: because this phase
bounds only the collapsed axes at zero thickness, ``bounding_box()`` returns
``lo == hi`` on a collapsed axis -- the slice position pulled into voxel
space, rounded by the same function -- and ``(-inf, +inf)`` on a displayed
one, clamped to the full extent.  So the output is byte-identical, computed
by the general machinery.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals._channel_appearance import ChannelAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance
from cellier.visuals._label_memory import InMemoryLabelsAppearance
from tests._v2 import bound

_WORLD = [("t", "time"), *spatial_axes("z", "y", "x")]
_SHAPE = (10, 20, 30, 40)


def _viewer(kind, transform_spec=None):
    """Add one visual of *kind* to a ``TZYX`` scene with one canvas."""
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_WORLD, dim="3d")
    controller.add_canvas(scene.id)
    if kind == "image":
        store = ImageMemoryStore(data=np.zeros(_SHAPE, dtype=np.float32))
        visual = controller.add_image(
            data=store,
            scene_id=scene.id,
            appearance=InMemoryImageAppearance(color_map="gray", clim=(0.0, 1.0)),
        )
    elif kind == "labels":
        store = LabelMemoryStore(data=np.zeros(_SHAPE, dtype=np.int32))
        visual = controller.add_labels(
            data=store, scene_id=scene.id, appearance=InMemoryLabelsAppearance()
        )
    else:
        store = ImageMemoryStore(data=np.zeros((3, 20, 30, 40), dtype=np.float32))
        visual = controller.add_multichannel_image(
            data=store,
            scene_id=scene.id,
            channel_axis=0,
            channels={
                0: ChannelAppearance(color_map="red", clim=(0.0, 1.0)),
                1: ChannelAppearance(color_map="green", clim=(0.0, 1.0)),
            },
        )
    if transform_spec is not None:
        visual.transform = bound(controller, scene.id, store, *transform_spec)
    return controller, scene, visual


def _plan(controller, scene, visual, *, with_region: bool):
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    selection = (
        controller._selections_for_scene(scene.id)[canvas_id] if with_region else None
    )
    requests = gfx.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        fov_y_rad=1.0,
        screen_height_px=100.0,
        dims_state=scene.dims.to_state(),
        selection=selection,
    )
    return [request.axis_selections for request in requests]


@pytest.mark.parametrize("kind", ["image", "labels", "multichannel"])
@pytest.mark.parametrize("slice_position", [0.0, 1.0, 2.5, 9.0, 99.0])
async def test_the_region_and_the_old_path_agree(kind, slice_position):
    controller, scene, visual = _viewer(kind)
    controller.update_slice_indices(scene.id, {0: slice_position})
    assert _plan(controller, scene, visual, with_region=True) == _plan(
        controller, scene, visual, with_region=False
    )


@pytest.mark.parametrize("kind", ["image", "labels"])
async def test_they_agree_under_an_awkward_transform(kind):
    """Design 3.7's numbers: ``T = 0.5 t + 0.25`` and ``Z = 2 z + 10``, chosen
    so the rounding step has something to do."""
    controller, scene, visual = _viewer(
        kind, ((0.5, 2.0, 0.5, 0.5), (0.25, 10.0, 0.0, 0.0))
    )
    controller.update_slice_indices(scene.id, {0: 1.0})
    with_region = _plan(controller, scene, visual, with_region=True)
    assert with_region == _plan(controller, scene, visual, with_region=False)
    # world T = 1 -> voxel t = 1.5 -> half-up -> 2, and the design says so.
    assert with_region == [(2, (0, 20), (0, 30), (0, 40))]


async def test_a_thickness_on_the_collapsed_axis_fetches_a_slab():
    """The extension design 3.7 promises: everything downstream of the region
    is untouched, and only the editor's thickness moves."""
    controller, scene, visual = _viewer("image")
    controller.update_slice_indices(scene.id, {0: 4.0})
    scene.dims.selection.thickness = {0: 1.5}
    assert _plan(controller, scene, visual, with_region=True) == [
        ((3, 7), (0, 20), (0, 30), (0, 40))
    ]


async def test_a_sheared_transform_is_rejected_when_it_is_assigned():
    """D7, caught one step earlier than the assembler.

    Today's in-memory path builds a zero-filled world point, writes the slice
    positions into it and inverts -- which answers "the ``t`` at world
    ``Z = Y = X = 0``", an arbitrary choice nothing states, and no error.  A
    sheared transform now fails when it is *assigned*, because the render
    spaces read the axis correspondence off the matrix and there is none.
    The assembler's own half-bounded guard is the second line of defence, and
    is covered in the assembler's tests.
    """
    from cellier.transform_v2 import AffineTransform

    controller, scene, visual = _viewer("image")
    store = controller.get_data_store(UUID(visual.data_store_id))
    matrix = np.eye(5)
    matrix[0, 1] = 0.1  # world T picks up a little of data z
    sheared = AffineTransform.from_matrix(
        matrix, store.data_coordinate_system, scene.dims.world_coordinate_system
    )
    with pytest.raises(Exception, match="one output axis per input axis"):
        visual.transform = sheared
