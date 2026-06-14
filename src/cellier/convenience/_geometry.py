"""Geometry utilities for computing world-space extents from a Viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from cellier.controller import CellierController
    from cellier.convenience._ortho_viewer import OrthoViewer
    from cellier.convenience._viewer import Viewer
    from cellier.scene.scene import Scene


def _axis_ranges_from_scene(
    controller: CellierController, scene: Scene
) -> dict[int, tuple[float, float]]:
    """Compute world-space axis ranges from the visuals in a single scene.

    Transforms the axis-aligned bounding box corners of each visual's backing
    data store from data space to world space and returns the per-axis union.
    Only stores exposing ``level_shapes`` (image and label stores) contribute;
    mesh, points, and lines visuals are skipped.

    Parameters
    ----------
    controller : CellierController
        Controller owning the data stores referenced by the scene's visuals.
    scene : Scene
        The scene whose visuals are inspected.

    Returns
    -------
    dict[int, tuple[float, float]]
        Mapping of axis index to ``(world_min, world_max)``.

    Raises
    ------
    ValueError
        If no qualifying visuals are found in the scene.
    """
    ndim = len(scene.dims.coordinate_system.axis_labels)

    world_mins = np.full(ndim, np.inf)
    world_maxs = np.full(ndim, -np.inf)
    found = False

    for visual_model in scene.visuals:
        store = controller.get_data_store(UUID(str(visual_model.data_store_id)))

        if not hasattr(store, "level_shapes"):
            continue

        shape = store.level_shapes[0]
        data_ndim = len(shape)

        origin = np.zeros((1, data_ndim), dtype=np.float64)
        far = np.array([[s - 1 for s in shape]], dtype=np.float64)
        corners = np.vstack([origin, far])
        world_corners = visual_model.transform.map_coordinates(
            corners.astype(np.float32)
        )

        world_mins = np.minimum(world_mins, world_corners.min(axis=0))
        world_maxs = np.maximum(world_maxs, world_corners.max(axis=0))
        found = True

    if not found:
        raise ValueError(
            "No visuals with known shapes found. "
            "Only image and label visuals contribute to axis ranges."
        )

    return {i: (float(world_mins[i]), float(world_maxs[i])) for i in range(ndim)}


def axis_ranges_from_viewer(viewer: Viewer) -> dict[int, tuple[float, float]]:
    """Compute world-space axis ranges by transforming each visual's bounding box.

    Walks every visual registered in the viewer's scene, transforms the
    axis-aligned bounding box corners of the backing data store from data
    space to world space using the visual's transform, and returns the union
    of all extents per world axis.

    Only visuals whose data store exposes a ``level_shapes`` property are
    included (image and label stores).  Mesh, points, and lines visuals are
    skipped.

    Parameters
    ----------
    viewer : Viewer
        The viewer to inspect.  Must have at least one visual with a
        ``level_shapes``-bearing data store.

    Returns
    -------
    dict[int, tuple[float, float]]
        Mapping of axis index to ``(world_min, world_max)``.

    Raises
    ------
    ValueError
        If no qualifying visuals are found.
    """
    return _axis_ranges_from_scene(viewer.controller, viewer.scene)


def axis_ranges_from_ortho(ortho: OrthoViewer) -> dict[int, tuple[float, float]]:
    """Compute world-space axis ranges for an :class:`OrthoViewer`.

    The four panels share their data stores, so the ranges are identical across
    panels.  The first panel that has qualifying visuals is used.

    Parameters
    ----------
    ortho : OrthoViewer
        The orthoviewer to inspect.  Must have at least one visual with a
        ``level_shapes``-bearing data store on some panel.

    Returns
    -------
    dict[int, tuple[float, float]]
        Mapping of axis index to ``(world_min, world_max)``.

    Raises
    ------
    ValueError
        If no qualifying visuals are found on any panel.
    """
    for scene in ortho.scenes.values():
        try:
            return _axis_ranges_from_scene(ortho.controller, scene)
        except ValueError:
            continue
    raise ValueError(
        "No visuals with known shapes found on any orthoviewer panel. "
        "Add an image or label visual before computing axis ranges."
    )
