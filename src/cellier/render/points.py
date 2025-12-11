"""Functions for constructing PyGFX objects from cellier models."""

from typing import Callable, Tuple

import numpy as np
import pygfx as gfx
from pygfx import PointsMaterial as GFXPointsMaterial

from cellier.models.visuals import PointsUniformAppearance, PointsVisual
from cellier.render.constants import cellier_to_gfx_coordinate_space
from cellier.transform import BaseTransform
from cellier.types import PointsDataResponse


def construct_pygfx_points_from_model(
    model: PointsVisual,
    empty_material: GFXPointsMaterial,
) -> Tuple[gfx.WorldObject, gfx.PointsMaterial]:
    """Make a PyGFX points object.

    This function dispatches to other constructor functions
    based on the material, etc. and returns a PyGFX mesh object.
    """
    # make the geometry
    # todo make initial slicing happen here or initialize with something more sensible

    # initialize with an dummy point coordinates
    # since we can't initialize an empty node.
    geometry = gfx.Geometry(
        positions=np.array([[0, 0, 0]], dtype=np.float32),
    )

    # make the material model
    appearance_model = model.appearance
    if isinstance(appearance_model, PointsUniformAppearance):
        size_space = cellier_to_gfx_coordinate_space[
            appearance_model.size_coordinate_space
        ]
        material = GFXPointsMaterial(
            size=appearance_model.size,
            size_space=size_space,
            color=appearance_model.color,
            size_mode="uniform",
            pick_write=model.pick_write,
        )
    else:
        raise TypeError(
            f"Unknown mesh material model type: {type(appearance_model)} in {model}"
        )
    return gfx.Points(
        geometry=geometry, material=empty_material, visible=model.appearance.visible
    ), material


class GFXPointsVisual:
    """PyGFX points node implementation.

    Note that PyGFX doesn't support empty WorldObjects, so we set
    transparent data when the slice is empty.
    """

    def __init__(self, model: PointsVisual):
        # This is the material given when the visual is "empty"
        # since pygfx doesn't support empty World Objects, we
        # initialize with a single point
        self._empty_material = GFXPointsMaterial(color=(0, 0, 0, 0))

        # make the pygfx materials
        self.node, self._material = construct_pygfx_points_from_model(
            model=model, empty_material=self._empty_material
        )

        # Flag that is set to True when there are no points to display.
        self._empty = True

    @property
    def material(self) -> GFXPointsMaterial:
        """The material object points."""
        return self._material

    @property
    def callback_handlers(self) -> list[Callable]:
        """Return the list of callback handlers for all nodes."""
        return [self.node.add_event_handler]

    def set_slice(self, slice_data: PointsDataResponse):
        """Set all the point coordinates."""
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
            coordinates = np.array([[0, 0, 0]], dtype=np.float32)

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

    def update_appearance(self, new_state: dict):
        """Update the appearance of the visual.

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
