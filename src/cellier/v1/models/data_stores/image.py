"""Classes for Image data stores."""

import time
from dataclasses import dataclass
from typing import Literal, Tuple, Union

import numpy as np
import zarr
from pydantic import ConfigDict, PrivateAttr, field_serializer, field_validator
from pydantic_core.core_schema import ValidationInfo

from cellier.models.data_stores.base_data_store import BaseDataStore, DataStoreSlice
from cellier.slicer.data_slice import RenderedImageDataSlice
from cellier.util.chunk import ImageDataStoreChunk, compute_chunk_corners_3d


@dataclass(frozen=True)
class ImageDataStoreSlice(DataStoreSlice):
    """Class containing data to slice a mesh data store.

    Parameters
    ----------
    displayed_dimensions : Union[Tuple[int, int, int], Tuple[int, int]]
        The indices of the displayed dimensions.
        The indices are ordered in their display order.
    scene_id : str
        The UID of the scene the visual is rendered in.
    visual_id : str
        The UID of the corresponding visual.
    resolution_level : int
        The resolution level to render where 0 is the highest resolution
        and high numbers correspond with more down sampling.
    point : tuple of floats
        Dims position in data coordinates for each dimension.
    margin_negative : tuple of floats
        Negative margin in data units of the slice for each dimension.
    margin_positive : tuple of floats
        Positive margin in data units of the slice for each dimension.
    """

    displayed_dimensions: Union[Tuple[int, int, int], Tuple[int, int]]
    point: Tuple[float, ...] = ()
    margin_negative: Tuple[float, ...] = ()
    margin_positive: Tuple[float, ...] = ()


class BaseImageDataStore(BaseDataStore):
    """The base class for all image data_stores."""

    name: str = "image_data_store"


class ImageMemoryStore(BaseImageDataStore):
    """Point data_stores store for arrays stored in memory."""

    data: np.ndarray

    # this is used for a discriminated union
    store_type: Literal["image_memory"] = "image_memory"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data", mode="before")
    @classmethod
    def coerce_to_ndarray_float32(cls, v: str, info: ValidationInfo):
        """Coerce to a float32 numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v.astype(np.float32)

    @field_serializer("data")
    def serialize_ndarray(self, array: np.ndarray, _info) -> list:
        """Coerce numpy arrays into lists for serialization."""
        return array.tolist()

    def get_slice(self, slice_data: ImageDataStoreSlice) -> RenderedImageDataSlice:
        """Get the data required to render a slice of the mesh.

        todo: generalize to oblique slicing
        """
        displayed_dimensions = list(slice_data.displayed_dimensions)

        slice_objects = [
            int(point_value)
            if (dimension_index not in displayed_dimensions)
            else slice(None)
            for dimension_index, point_value in enumerate(slice_data.point)
        ]

        return RenderedImageDataSlice(
            scene_id=slice_data.scene_id,
            visual_id=slice_data.visual_id,
            resolution_level=slice_data.resolution_level,
            data=self.data[tuple(slice_objects)],
        )


class MockLatentImageStore(BaseImageDataStore):
    """Point data_stores store for arrays stored in memory."""

    data: np.ndarray

    # the time to pause for each slice
    slice_time: float = 1

    # this is used for a discriminated union
    store_type: Literal["mock_latent_image"] = "mock_latent_image"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data", mode="before")
    @classmethod
    def coerce_to_ndarray_float32(cls, v: str, info: ValidationInfo):
        """Coerce to a float32 numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v.astype(np.float32)

    @field_serializer("data")
    def serialize_ndarray(self, array: np.ndarray, _info) -> list:
        """Coerce numpy arrays into lists for serialization."""
        return array.tolist()

    def get_slice(self, slice_data: ImageDataStoreSlice) -> RenderedImageDataSlice:
        """Get the data required to render a slice of the mesh.

        todo: generalize to oblique slicing
        """
        displayed_dimensions = list(slice_data.displayed_dimensions)

        time.sleep(self.slice_time)

        slice_objects = [
            int(point_value)
            if (dimension_index not in displayed_dimensions)
            else slice(None)
            for dimension_index, point_value in enumerate(slice_data.point)
        ]

        return RenderedImageDataSlice(
            scene_id=slice_data.scene_id,
            visual_id=slice_data.visual_id,
            resolution_level=slice_data.resolution_level,
            data=self.data[tuple(slice_objects)],
        )


class MultiScaleImageZarrStore(BaseImageDataStore):
    """Data store for a multiscale zarr image."""

    root_path: str
    scale_paths: list[str]
    scales: list[tuple[float, float, float]]
    translations: list[tuple[float, float, float]]

    # this will get populated with the arrays
    _root_group: zarr.Group = PrivateAttr()
    _arrays: list[zarr.Array] = PrivateAttr()
    _min_voxel_size_local: np.ndarray = PrivateAttr()
    _chunk_corners: list[np.ndarray] = PrivateAttr()
    _chunks_shapes: list[tuple[int, ...]]

    # this is used for a discriminated union
    store_type: Literal["multiscale_image_zarr"] = "multiscale_image_zarr"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # pre-open store the zarr array objects
        self._root_group = zarr.open(self.root_path)
        self._arrays = [self._root_group[array_path] for array_path in self.scale_paths]

        # get the voxel size of each scale in the local coordinates
        self._min_voxel_size_local = np.array(
            [np.min(scale_level) for scale_level in self.scales]
        )

        # get the corners for each chunk at each scale
        self._chunk_corners = [
            compute_chunk_corners_3d(
                array_shape=np.asarray(array.shape),
                chunk_shape=np.asarray(array.chunks),
            )
            for array in self.arrays
        ]

        self._chunk_shapes = [array.chunks for array in self.arrays]

    @property
    def n_scales(self) -> int:
        """The number of scales in this image."""
        return len(self._arrays)

    @property
    def arrays(self) -> list[zarr.Array]:
        """The Zarr arrays of each scale in this image.

        The arrays are ordered from the highest resolution to the lowest resolution.
        """
        return self._arrays

    @property
    def chunk_corners(self) -> list[np.ndarray]:
        """The corner of each chunks at each scale level.

        The indices in the list are the scale level.
        """
        return self._chunk_corners

    @property
    def chunk_shapes(self) -> list[tuple[int, ...]]:
        """The shape of the chunk at each scale level.

        The indices in the list are the scale level.
        """
        return self._chunk_shapes

    def get_slice(self, slice_data: ImageDataStoreChunk) -> RenderedImageDataSlice:
        """Get the data required to render a slice of the mesh.

        todo: generalize to oblique slicing
        """
        # displayed_dimensions = list(slice_data.displayed_dimensions)
        #
        #
        #
        # slice_objects = [
        #     int(point_value)
        #     if (dimension_index not in displayed_dimensions)
        #     else slice(None)
        #     for dimension_index, point_value in enumerate(slice_data.point)
        # ]
        scale_array = self._arrays[slice_data.resolution_level]
        slice_objects = [
            slice(start, end)
            for start, end in zip(
                slice_data.array_coordinate_start, slice_data.array_coordinate_end
            )
        ]

        return RenderedImageDataSlice(
            scene_id=slice_data.scene_id,
            visual_id=slice_data.visual_id,
            resolution_level=slice_data.resolution_level,
            data=scale_array[tuple(slice_objects)],
            texture_start_index=slice_data.texture_coordinate_start,
        )

    def determine_scale_from_frustum(
        self,
        frustum_corners: np.ndarray,
        width_logical: int,
        height_logical: int,
        method: str = "logical_pixel_size",
    ) -> int:
        """Determine which scale to render at based on the view frustum."""
        if method == "logical_pixel_size":
            near_plane = frustum_corners[0]
            width_local = np.linalg.norm(near_plane[1, :] - near_plane[0, :])
            height_local = np.linalg.norm(near_plane[3, :] - near_plane[0, :])

            return self._select_scale_by_logical_voxel_size(
                frustum_width_local=width_local,
                frustum_height_local=height_local,
                width_logical=width_logical,
                height_logical=height_logical,
            )

        else:
            raise ValueError(f"Unknown method: {method}.")

    def _select_scale_by_logical_voxel_size(
        self,
        frustum_width_local: float,
        frustum_height_local: float,
        width_logical: int,
        height_logical: int,
    ) -> int:
        """Select the scale based on the size of the logical voxel.

        This method tries to select a scale where the size of the voxel
        is closest to the one logical pixel.
        """
        # get the smallest size of the logical pixels
        logical_pixel_width_local = frustum_width_local / width_logical
        logical_pixel_height_local = frustum_height_local / height_logical
        logical_pixel_local = min(logical_pixel_width_local, logical_pixel_height_local)

        pixel_size_difference = self._min_voxel_size_local - logical_pixel_local

        for scale_index in reversed(range(self.n_scales)):
            if pixel_size_difference[scale_index] <= 0:
                selected_level_index = min(self.n_scales - 1, scale_index + 1)
                return selected_level_index

        # if none work, return the highest resolution
        return 0
