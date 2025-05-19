"""PyGFX implementations of line visuals."""

from typing import Callable

import numpy as np
import pygfx as gfx

from cellier.models.visuals import LinesUniformMaterial, LinesVisual
from cellier.render.constants import cellier_to_gfx_coordinate_space
from cellier.types import LinesDataResponse


def construct_pygfx_lines_from_model(
    model: LinesVisual, empty_material: gfx.LineMaterial
):
    """Make a PyGFX line object.

    This function dispatches to other constructor functions
    based on the material, etc. and returns a PyGFX mesh object.
    """
    # initialize with dummy coordinates
    # since we can't initialize an empty node.
    geometry = gfx.Geometry(
        positions=np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float32),
    )
    material_model = model.material
    if isinstance(material_model, LinesUniformMaterial):
        size_space = cellier_to_gfx_coordinate_space[
            material_model.size_coordinate_space
        ]
        material = gfx.LineSegmentMaterial(
            thickness_space=size_space,
            thickness=material_model.size,
            color=material_model.color,
            opacity=material_model.opacity,
            pick_write=model.pick_write,
        )
    else:
        raise TypeError(
            f"Unknown mesh material model type: {type(material_model)} in {model}"
        )
    return gfx.Line(geometry=geometry, material=empty_material), material


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

        new_geometry = gfx.Geometry(positions=coordinates)
        self.node.geometry = new_geometry

        if was_empty and not self._empty:
            # if this is the first data after the layer
            # was empty, set the material
            self.node.material = self.material
        elif not was_empty and self._empty:
            # if the layer has become empty, set the material
            self.node.material = self._empty_material
