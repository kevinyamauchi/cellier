"""Visual model for chunked, multiscale 3D image data."""

from typing import Literal

from cellier.models.visuals.base import BaseVisual
from cellier.models.visuals.image import ImageAppearance


class ChunkedImageVisual(BaseVisual):
    """Model for a chunked, multiscale 3D image visual.

    This visual type is designed to work with
    :class:`~cellier.models.data_stores.chunked_image.ChunkedImageStore`.
    The store manages frustum-culled chunk selection, async loading, and
    an LRU cache; the matching render node
    (:class:`~cellier.render.chunked_image.GFXChunkedImageNode`) manages a
    GPU texture atlas and uploads chunks as they arrive.

    Parameters
    ----------
    name : str
        Human-readable label for this visual.
    data_store_id : str
        ID of the :class:`~cellier.models.data_stores.chunked_image.ChunkedImageStore`
        that supplies voxel data.
    appearance : ImageAppearance
        Colour-map and visibility settings.
    """

    data_store_id: str
    appearance: ImageAppearance

    # discriminated union key
    visual_type: Literal["chunked_image"] = "chunked_image"
