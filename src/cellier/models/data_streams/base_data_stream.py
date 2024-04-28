"""Base class for all data_stores streams."""

from abc import ABC, abstractmethod
from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field

from cellier.slicer.transform import BaseTransform
from cellier.slicer.world_slice import BaseWorldSlice


class BaseDataStream(EventedModel, ABC):
    """Base class for all data_stores streams.

    Parameters
    ----------
    id : str
        The unique identifier for the data store.
        The default value is a UUID4 generated hex string.

    Attributes
    ----------
    id : str
        The unique identifier for this data stream instance.
    """

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)

    @abstractmethod
    def get_data_slice(
        self,
        world_slice: BaseWorldSlice,
        visual_id: str,
        request_id: str,
        data_to_world_transform: BaseTransform,
    ):
        """Get a data slice from a world slice request.

        Parameters
        ----------
        world_slice : BaseWorldSlice
            The data of the world slice to get from the data stream.
            The coordinates are in world coordinates.
        visual_id : str
            The unique identifier for which visual this data slice
            will be sent to.
        request_id : str
            The unique identifier for this request.
        data_to_world_transform : BaseTransform
            The transformation from the data coordinates to the world coordinates.
        """
        raise NotImplementedError
