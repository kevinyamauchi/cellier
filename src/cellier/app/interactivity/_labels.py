"""Class and utilities to manage labels painting.

This is modified from napari's Labels layer.
https://github.com/napari/napari/blob/main/napari/layers/labels/_labels_mouse_bindings.py
https://github.com/napari/napari/blob/main/napari/layers/labels/_labels_utils.py

License for napari:
BSD 3-Clause License

Copyright (c) 2018, Napari
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from enum import Enum
from functools import lru_cache

import numpy as np

from cellier.events import MouseCallbackData
from cellier.models.data_stores import ImageMemoryStore
from cellier.models.visuals import MultiscaleLabelsVisual
from cellier.types import MouseButton, MouseModifiers


def _get_shape_and_dims_to_paint(
    image_shape: tuple[int, ...], ordered_dims: tuple[int, ...], n_dim_paint: int
) -> tuple[list, list]:
    """Get the shape of the data and the dimensions to paint.

    Parameters
    ----------
    image_shape : tuple[int, ...]
        The shape of the image data.
    ordered_dims : tuple[int, ...]
        The ordered dimensions of the image data.
        These should be ordered such at the displayed dimensions
        are last.
    n_dim_paint : int
        The number of dimensions to paint.

    Returns
    -------
    image_shape_displayed : list
        The shape of the image in the displayed dimensions.

    dims_to_paint : list
        The indices of the dimensions to paint.
    """
    ordered_dims = np.asarray(ordered_dims)
    dims_to_paint = sorted(ordered_dims[n_dim_paint:])
    image_shape_nD = list(image_shape)

    image_shape_displayed = [image_shape_nD[i] for i in dims_to_paint]

    return image_shape_displayed, dims_to_paint


@lru_cache(maxsize=64)
def sphere_indices(radius, scale):
    """Generate centered indices within circle or n-dim ellipsoid.

    Parameters
    ----------
    radius : float
        Radius of circle/sphere
    scale : tuple of float
        The scaling to apply to the sphere along each axis

    Returns
    -------
    mask_indices : array
        Centered indices within circle/sphere
    """
    ndim = len(scale)
    abs_scale = np.abs(scale)
    scale_normalized = np.asarray(abs_scale, dtype=float) / np.min(abs_scale)
    # Create multi-dimensional grid to check for
    # circle/membership around center
    r_normalized = radius / scale_normalized + 0.5
    slices = [slice(-int(np.ceil(r)), int(np.floor(r)) + 1) for r in r_normalized]

    indices = np.mgrid[slices].T.reshape(-1, ndim)
    distances_sq = np.sum((indices * scale_normalized) ** 2, axis=1)
    # Use distances within desired radius to mask indices in grid
    mask_indices = indices[distances_sq <= radius**2].astype(int)

    return mask_indices


def interpolate_painting_coordinates(old_coord, new_coord, brush_size):
    """Interpolates coordinates depending on brush size.

    Useful for ensuring painting is continuous in labels layer.

    Parameters
    ----------
    old_coord : np.ndarray, 1x2
        Last position of cursor.
    new_coord : np.ndarray, 1x2
        Current position of cursor.
    brush_size : float
        Size of brush, which determines spacing of interpolation.

    Returns
    -------
    coords : np.array, Nx2
        List of coordinates to ensure painting is continuous
    """
    if old_coord is None:
        old_coord = new_coord
    if new_coord is None:
        new_coord = old_coord
    num_step = round(
        max(abs(np.array(new_coord) - np.array(old_coord))) / brush_size * 4
    )
    coords = [
        np.linspace(old_coord[i], new_coord[i], num=int(num_step + 1))
        for i in range(len(new_coord))
    ]
    coords = np.stack(coords).T
    if len(coords) > 1:
        coords = coords[1:]

    return coords


class LabelsPaintingMode(Enum):
    """Enum for the different modes of painting labels.

    Attributes
    ----------
    NONE : str
        No painting.
    PAINT : str
        Paint the labels.
    ERASE : str
        Erase the labels.
    FILL : str
        Fill connected components with the same label.
    """

    NONE = "none"
    PAINT = "paint"
    ERASE = "erase"
    FILL = "fill"


class LabelsPaintingManager:
    """Class to manage the painting of labels data.

    Parameters
    ----------
    model : MultiscaleLabelsVisual
        The model for the labels visual to be painted.
        Currently, only labels with a single scale are supported.
    data_store : ImageMemoryStore
        The data store for the labels visual to be painted.
    """

    def __init__(
        self,
        model: MultiscaleLabelsVisual,
        data_store: ImageMemoryStore,
        mode: LabelsPaintingMode = LabelsPaintingMode.NONE,
    ):
        if len(MultiscaleLabelsVisual.downscale_factors) != 1:
            raise NotImplementedError("Only single scale labels are supported.")

        self._model = model
        self._data = data_store
        self._mode = mode

        # hack - these properties should come from the mouse event
        # todo fix
        self._ordered_dims = (0, 1, 2)
        self._n_dim_paint = 2

        # Currently, the background value must be 0.
        # We may consider changing this in the future.
        self._background_value = 0

        # This is the value that will be painted.
        self._value_to_paint = 2

    @property
    def value_to_paint(self) -> int:
        """Returns the value that will be painted."""
        return self._value_to_paint

    @value_to_paint.setter
    def value_to_paint(self, value: int):
        """Sets the value that will be painted."""
        self._value_to_paint = value

    @property
    def background_val(self) -> int:
        """Returns the background value.

        Currently, this must be 0 and thus cannot be set.
        We may consider changing this in the future.
        """
        return self._background_value

    def paint(self, coord, new_label, refresh=True):
        """Paint over existing labels with a new label.

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in image coordinates.
        new_label : int
            Value of the new label to be filled in.
        refresh : bool
            Whether to refresh view slice or not. Set to False to batch paint
            calls.
        """
        shape, dims_to_paint = _get_shape_and_dims_to_paint(
            image_shape=self._data.data.shape,
            ordered_dims=self._ordered_dims,
            n_dim_paint=self._n_dim_paint,
        )
        paint_scale = np.array([self.scale[i] for i in dims_to_paint], dtype=float)

        slice_coord = [int(np.round(c)) for c in coord]
        if self.n_edit_dimensions < self.ndim:
            coord_paint = [coord[i] for i in dims_to_paint]
        else:
            coord_paint = coord

        # Ensure circle doesn't have spurious point
        # on edge by keeping radius as ##.5
        radius = np.floor(self.brush_size / 2) + 0.5
        mask_indices = sphere_indices(radius, tuple(paint_scale))

        mask_indices = mask_indices + np.round(np.array(coord_paint)).astype(int)

        self._paint_indices(
            mask_indices, new_label, shape, dims_to_paint, slice_coord, refresh
        )

    def mouse_press(self, event: MouseCallbackData):
        """Handle mouse press events.

        This currently only works for 2D and doesn't handle transforms
        """
        if (event.button != MouseButton.LEFT) or (
            MouseModifiers.SHIFT in event.modifiers
        ):
            # only paint when LMB is pressed. also do not paint when shift is pressed.
            return

        coordinates = event.coordinate
        if self._mode == LabelsPaintingMode.ERASE:
            new_label = self.background_value
        else:
            new_label = self.value_to_paint

        # on press
        # with layer.block_history():
        self._draw(new_label, coordinates, coordinates)
        yield

        last_cursor_coord = coordinates
        # on move
        while event.button == MouseButton.LEFT:
            coordinates = event.coordinate
            if coordinates is not None or last_cursor_coord is not None:
                self._draw(new_label, last_cursor_coord, coordinates)
            last_cursor_coord = coordinates
            yield

    def _draw(
        self, label_value: int, last_cursor_coordinate, current_cursor_coordinate
    ):
        """Draw the label value on the data store.

        Parameters
        ----------
        label_value : int
            The label value to draw.
        last_cursor_coordinate : tuple[int, int, int]
            The last cursor coordinate.
        current_cursor_coordinate : tuple[int, int, int]
            The current cursor coordinate.
        """
        if current_cursor_coordinate is None:
            return
        interp_coord = interpolate_painting_coordinates(
            last_cursor_coordinate, current_cursor_coordinate, self.brush_size
        )
        for c in interp_coord:
            if (
                self._slice_input.ndisplay == 3
                and self.data[tuple(np.round(c).astype(int))] == 0
            ):
                continue
            if self._mode in [LabelsPaintingMode.PAINT, LabelsPaintingMode.ERASE]:
                self.paint(c, label_value, refresh=False)
            elif self._mode == LabelsPaintingMode.FILL:
                self.fill(c, label_value, refresh=False)
        # self._partial_labels_refresh()
