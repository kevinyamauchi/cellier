"""Base class for all data_stores streams."""

from abc import ABC, abstractmethod
from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field

from cellier.slicer.data_slice import DataSliceRequest


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
    def get_data_store_slice(self, slice_request: DataSliceRequest):
        """Get slice object to get the requested world data slice from the data store.

        Parameters
        ----------
        slice_request : DataSliceRequest
            The requested data slice to generate the data store slice from.
        """
        raise NotImplementedError
