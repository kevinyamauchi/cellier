"""PyGFX implementations of line visuals."""

from typing import Callable

import numpy as np
import pygfx as gfx

from cellier.models.visuals import (
    LinesUniformAppearance,
    LinesVertexColorAppearance,
    LinesVisual,
)
from cellier.render.constants import cellier_to_gfx_coordinate_space
from cellier.transform import BaseTransform
from cellier.types import LinesDataResponse


def construct_pygfx_lines_from_model(
    model: LinesVisual, empty_material: gfx.LineMaterial
):
    """Make a PyGFX line object.

    This function dispatches to other constructor functions
    based on the material, etc. and returns a PyGFX mesh object.
    """
    appearance_model = model.appearance
    if isinstance(appearance_model, LinesUniformAppearance):
        # initialize with dummy coordinates
        # since we can't initialize an empty node.
        geometry = gfx.Geometry(
            positions=np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float32),
        )

        # make the appearance
        size_space = cellier_to_gfx_coordinate_space[
            appearance_model.size_coordinate_space
        ]
        appearance = gfx.LineSegmentMaterial(
            thickness_space=size_space,
            thickness=appearance_model.size,
            color=appearance_model.color,
            opacity=appearance_model.opacity,
            color_mode="uniform",
            pick_write=model.pick_write,
        )
    elif isinstance(appearance_model, LinesVertexColorAppearance):
        # initialize with dummy coordinates
        # since we can't initialize an empty node.
        geometry = gfx.Geometry(
            positions=np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float32),
            colors=np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.float32),
        )

        # make the appearance
        size_space = cellier_to_gfx_coordinate_space[
            appearance_model.size_coordinate_space
        ]
        appearance = gfx.LineSegmentMaterial(
            thickness_space=size_space,
            thickness=appearance_model.size,
            opacity=appearance_model.opacity,
            color_mode="vertex",
            pick_write=model.pick_write,
        )
    else:
        raise TypeError(
            f"Unknown mesh material model type: {type(appearance_model)} in {model}"
        )
    return gfx.Line(
        geometry=geometry, material=empty_material, visible=model.appearance.visible
    ), appearance


class GFXLinesVisual:
    """PyGFX lines node implementation.

    Note that PyGFX doesn't support empty WorldObjects, so we set
    transparent data when the slice is empty.
    """

    def __init__(self, model: LinesVisual):
        # This is the material given when the visual is "empty"
        # since pygfx doesn't support empty World Objects, we
        # initialize with a single line
        self._empty_material = gfx.LineMaterial(color=(0, 0, 0, 0))

        self.node, self._material = construct_pygfx_lines_from_model(
            model, self._empty_material
        )

        # Flag that is set to True when there are no points to display.
        self._empty = True

    @property
    def material(self) -> gfx.LineMaterial:
        """The material object points."""
        return self._material

    @property
    def callback_handlers(self) -> list[Callable]:
        """Return the list of callback handlers for all nodes."""
        return [self.node.add_event_handler]

    def set_slice(self, slice_data: LinesDataResponse):
        """Set the slice data for the lines."""
        coordinates = slice_data.data

        # check if the layer was empty
        was_empty = self._empty

        if coordinates.shape[1] == 2:
            # pygfx expects 3D points
            n_points = coordinates.shape[0]
            zeros_column = np.zeros((n_points, 1), dtype=np.float32)
            coordinates = np.column_stack((coordinates, zeros_column))

        if coordinates.shape[0] == 0:
            # coordinates must not be empty
            # todo do something smarter?
            coordinates = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float32)

            # set the empty flag
            self._empty = True
        else:
            # There is data to set, so the node is not empty
            self._empty = False

        if slice_data.colors is None:
            new_geometry = gfx.Geometry(positions=coordinates)
        else:
            if slice_data.colors.shape[0] == 0:
                # if the colors are empty, we set them to transparent
                colors = np.zeros_like(coordinates, dtype=np.float32)
            else:
                colors = slice_data.colors.astype(np.float32)
            new_geometry = gfx.Geometry(
                positions=coordinates,
                colors=colors,
            )
        self.node.geometry = new_geometry

        if was_empty and not self._empty:
            # if this is the first data after the layer
            # was empty, set the material
            self.node.material = self.material
        elif not was_empty and self._empty:
            # if the layer has become empty, set the material
            self.node.material = self._empty_material

    def update_appearance(self, new_state: dict):
        """Update the state of the visual.

        This is generally used as a callback for when
        the visual model updates.
        """
        if "visible" in new_state:
            # update the visibility
            self.node.visible = new_state["visible"]

    def set_transform(self, transform: BaseTransform):
        """Set the local transform of the node.

        Parameters
        ----------
        transform : np.ndarray
            The 4x4 affine transform matrix to set.
        """
        # set the local transform
        self.node.local.matrix = transform.matrix
