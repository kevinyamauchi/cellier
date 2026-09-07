"""Base class for data stores."""

from __future__ import annotations

import uuid
from typing import Annotated, ClassVar
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field

from cellier.data._dataset_info import DatasetInfo, RowSection


class BaseDataStore(EventedModel):
    """The base class for all DataStores.

    Parameters
    ----------
    id : UUID4
        The unique identifier for the data store.
        The default value is a UUID4 generated hex string.
    name : str
        The name of the data store.

    Attributes
    ----------
    id : str
        The unique identifier for the data store.
    """

    # store a UUID to identify this specific scene.
    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    name: str = "data store"

    # ── Self-description ────────────────────────────────────────────────

    DATASET_INFO_LABEL: ClassVar[str] = "data store"
    """What this kind of store calls itself in the ``Store type`` row.

    A human-readable name rather than the ``store_type`` discriminator: the
    row is read by a person, and ``"in-memory points"`` says more than
    ``"points_memory"``.
    """

    def dataset_info(self) -> DatasetInfo:
        """Describe what this store holds, for display in a dataset-info widget.

        Returns the store's identity only.  Each concrete store overrides
        this to append what is specific to it -- shapes and scale levels for
        an image, node and edge counts for a graph -- so a new store type
        gains a populated widget by implementing one method, with nothing to
        register in the GUI layer.

        Every implementation is **cheap**: it reads metadata the store
        already holds and never triggers a read of the underlying array.
        Statistics that would require one (an image's value range, a label
        image's unique labels) are reported only where the data is already
        resident in memory.

        Returns
        -------
        DatasetInfo
            The sections to draw, in display order.
        """
        return DatasetInfo(sections=[RowSection(None, self._identity_rows())])

    def _identity_rows(self) -> list[tuple[str, str]]:
        """The ``Name``/``Store type`` rows every store's block opens with."""
        return [
            ("Name", self.name),
            ("Store type", self.DATASET_INFO_LABEL),
        ]
