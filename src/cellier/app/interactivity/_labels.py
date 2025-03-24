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

import inspect
from enum import Enum
from functools import lru_cache
from typing import Iterator

import numpy as np
import numpy.typing as npt

from cellier.models.data_stores import ImageMemoryStore
from cellier.models.scene import OrthographicCamera, PerspectiveCamera
from cellier.models.visuals import MultiscaleLabelsVisual
from cellier.types import MouseButton, MouseCallbackData, MouseEventType, MouseModifiers


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
    dims_to_paint = sorted(ordered_dims[-n_dim_paint:])
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


def indices_in_shape(idxs, shape):
    """Return idxs after filtering out indices that are not in given shape.

    Parameters
    ----------
    idxs : tuple of array of int, or 2D array of int
        The input coordinates. These should be in one of two formats:

        - a tuple of 1D arrays, as for NumPy fancy indexing, or
        - a 2D array of shape (ncoords, ndim), as a list of coordinates

    shape : tuple of int
        The shape in which all indices must fit.

    Returns
    -------
    idxs_filtered : tuple of array of int, or 2D array of int
        The subset of the input idxs that falls within shape.

    Examples
    --------
    >>> idxs0 = (np.array([5, 45, 2]), np.array([6, 5, -5]))
    >>> indices_in_shape(idxs0, (10, 10))
    (array([5]), array([6]))
    >>> idxs1 = np.transpose(idxs0)
    >>> indices_in_shape(idxs1, (10, 10))
    array([[5, 6]])
    """
    np_index = isinstance(idxs, tuple)
    if np_index:  # normalize to 2D coords array
        idxs = np.transpose(idxs)
    keep_coords = np.logical_and(
        np.all(idxs >= 0, axis=1), np.all(idxs < np.array(shape), axis=1)
    )
    filtered = idxs[keep_coords]
    if np_index:  # convert back to original format
        filtered = tuple(filtered.T)
    return filtered


def _arraylike_short_names(obj) -> Iterator[str]:
    """Yield all the short names of an array-like or its class."""
    type_ = type(obj) if not inspect.isclass(obj) else obj
    for base in type_.mro():
        yield f'{base.__module__.split(".", maxsplit=1)[0]}.{base.__name__}'


def _is_array_type(array: npt.ArrayLike, type_name: str) -> bool:
    """Checks if an array-like is of the type described by a short name.

    This is useful when you want to check the type of array-like quickly without
    importing its package, which might take a long time.

    Parameters
    ----------
    array
        The array-like object.
    type_name : str
        The short name of the type to test against
        (e.g. 'numpy.ndarray', 'xarray.DataArray').

    Returns
    -------
    True if the array is associated with the type name.
    """
    return type_name in _arraylike_short_names(array)


def _coerce_indices_for_vectorization(array, indices: list) -> tuple:
    """Coerces indices so that they can be used for vectorized indexing."""
    if _is_array_type(array, "xarray.DataArray"):
        # Fix indexing for xarray if necessary
        # See http://xarray.pydata.org/en/stable/indexing.html#vectorized-indexing
        # for difference from indexing numpy
        try:
            import xarray as xr
        except ModuleNotFoundError:
            pass
        else:
            return tuple(xr.DataArray(i) for i in indices)
    return tuple(indices)


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
    labels_model : MultiscaleLabelsVisual
        The model for the labels visual to be painted.
        Currently, only labels with a single scale are supported.
    data_store : ImageMemoryStore
        The data store for the labels visual to be painted.
    """

    def __init__(
        self,
        labels_model: MultiscaleLabelsVisual,
        camera_model: PerspectiveCamera | OrthographicCamera,
        data_store: ImageMemoryStore,
        mode: LabelsPaintingMode = LabelsPaintingMode.PAINT,
    ):
        if len(labels_model.downscale_factors) != 1:
            raise NotImplementedError("Only single scale labels are supported.")

        self._labels_model = labels_model
        self._camera_model = camera_model
        self._data = data_store

        # we set the mode to None to initialize.
        # it will get set by the actual value at the end of the init.
        self._mode = LabelsPaintingMode.NONE

        # store the state of the camera before it gets changed by the mode.
        self._pre_paint_camera_controller_enabled = (
            self._camera_model.controller.enabled
        )

        # hack - these properties should come from the mouse event
        # todo fix
        self._ordered_dims = (0, 1, 2)
        self._n_dim_paint = 2
        self._ndisplay = 2
        self._ndim = 3
        self._scale = (1, 1, 1)

        # the number of dimensions to paint on
        self._n_edit_dimensions = 2

        # Currently, the background value must be 0.
        # We may consider changing this in the future.
        self._background_value = 0

        # This is the value that will be painted.
        self._value_to_paint = 2

        # the size of the brush to paint with
        self._brush_size = 2

        # state variables for painting
        self._last_coordinate = None
        self._dragging = False

        # set the mode
        self.mode = mode

    @property
    def mode(self) -> LabelsPaintingMode:
        """Returns the current mode of painting."""
        return self._mode

    @mode.setter
    def mode(self, mode: LabelsPaintingMode):
        """Sets the current mode of painting."""
        if not isinstance(mode, LabelsPaintingMode):
            mode = LabelsPaintingMode(mode)

        if mode == self._mode:
            # if the values hasn't changed, don't do anything
            return

        if mode != LabelsPaintingMode.NONE:
            # turn off the camera controller so it doesn't move while painting
            self._pre_paint_camera_controller_enabled = (
                self._camera_model.controller.enabled
            )
            self._camera_model.controller.enabled = False

        else:
            # turn the camera controller back on
            self._camera_model.controller.enabled = (
                self._pre_paint_camera_controller_enabled
            )

        self._mode = mode

    @property
    def brush_size(self) -> int:
        """Returns the size of the brush."""
        return self._brush_size

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
        paint_scale = np.array([self._scale[i] for i in dims_to_paint], dtype=float)

        slice_coord = [int(np.round(c)) for c in coord]
        if self._n_edit_dimensions < self._ndim:
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

    def _on_mouse_press(self, event: MouseCallbackData):
        """Paint using the mouse press events."""
        if (
            (event.button == MouseButton.LEFT)
            and (MouseModifiers.SHIFT not in event.modifiers)
            and (self.mode != LabelsPaintingMode.NONE)
        ):
            if self._mode == LabelsPaintingMode.ERASE:
                new_label = self.background_value
            else:
                new_label = self.value_to_paint

            # todo fix hack
            coordinates = np.zeros((3,))
            coordinates[0] = event.coordinate[0]
            coordinates[1] = event.coordinate[2]
            coordinates[2] = event.coordinate[1]
            if self._mode == LabelsPaintingMode.ERASE:
                new_label = self.background_value
            else:
                new_label = self.value_to_paint

            # on press
            # with layer.block_history():

            if self._last_coordinate is None:
                self._last_coordinate = coordinates

            self._draw(
                label_value=new_label,
                last_cursor_coordinate=self._last_coordinate,
                current_cursor_coordinate=coordinates,
            )
            self._last_coordinate = coordinates
            self._dragging = True

            if event.type == MouseEventType.RELEASE:
                # end of drag
                self._dragging = False
                self._last_coordinate = None
        else:
            if self._dragging:
                # todo fix hack
                coordinates = np.zeros((3,))
                coordinates[0] = event.coordinate[0]
                coordinates[1] = event.coordinate[2]
                coordinates[2] = event.coordinate[1]

                if np.linalg.norm(coordinates - self._last_coordinate) < 1:
                    # don't paint if the cursor hasn't moved
                    return

                if self._mode == LabelsPaintingMode.ERASE:
                    new_label = self.background_value
                else:
                    new_label = self.value_to_paint
                self._draw(
                    label_value=new_label,
                    last_cursor_coordinate=self._last_coordinate,
                    current_cursor_coordinate=coordinates,
                )

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
                self._ndisplay == 3
                and self.data.data[tuple(np.round(c).astype(int))] == 0
            ):
                continue
            if self._mode in [LabelsPaintingMode.PAINT, LabelsPaintingMode.ERASE]:
                self.paint(c, label_value, refresh=False)
            elif self._mode == LabelsPaintingMode.FILL:
                self.fill(c, label_value, refresh=False)
        # self._partial_labels_refresh()

    def _paint_indices(
        self,
        mask_indices,
        new_label,
        shape,
        dims_to_paint,
        slice_coord=None,
        refresh=True,
    ):
        """Paint over existing labels with a new label, using the selected mask indices.

        Depending on the dims to paint it can either be only on the visible slice
        or in all n dimensions.

        Parameters
        ----------
        mask_indices : numpy array of integer coordinates
            Mask to paint represented by an array of its coordinates.
        new_label : int
            Value of the new label to be filled in.
        shape : list
            The label data shape upon which painting is performed.
        dims_to_paint : list
            List of dimensions of the label data that are used for painting.
        refresh : bool
            Whether to refresh view slice or not. Set to False to batch paint
            calls.
        slice_coord : tuple[int, ...] | None
            The slice coordinates for the array to be painted.
        """
        dims_not_painted = sorted(self._ordered_dims[: -self._n_edit_dimensions])
        # discard candidate coordinates that are out of bounds
        mask_indices = indices_in_shape(mask_indices, shape)

        # Transfer valid coordinates to slice_coord,
        # or expand coordinate if 3rd dim in 2D image
        slice_coord_temp = list(mask_indices.T)
        if self._n_edit_dimensions < self._ndim:
            for j, i in enumerate(dims_to_paint):
                slice_coord[i] = slice_coord_temp[j]
            for i in dims_not_painted:
                slice_coord[i] = slice_coord[i] * np.ones(
                    mask_indices.shape[0], dtype=int
                )
        else:
            slice_coord = slice_coord_temp

        slice_coord = _coerce_indices_for_vectorization(self._data.data, slice_coord)

        # slice coord is a tuple of coordinate arrays per dimension
        # subset it if we want to only paint into background/only erase
        # current label
        # if self.preserve_labels:
        #     if new_label == self.colormap.background_value:
        #         keep_coords = self.data[slice_coord] == self.selected_label
        #     else:
        #         keep_coords = (
        #                 self.data[slice_coord] == self.colormap.background_value
        #         )
        #     slice_coord = tuple(sc[keep_coords] for sc in slice_coord)

        self._data.data[slice_coord] = new_label
        self._data.events.data.emit()
