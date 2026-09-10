"""Base class for data stores."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, ClassVar
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field, model_validator

from cellier.data._axes import (
    build_axes,
    data_coordinate_system,
    identity_transform,
)
from cellier.data._dataset_info import DatasetInfo, RowSection
from cellier.transform_v2 import (  # noqa: TC001
    AffineTransform,
    DataCoordinateSystem,
)


class BaseDataStore(EventedModel):
    """The base class for all DataStores.

    Parameters
    ----------
    id : UUID4
        The unique identifier for the data store.
        The default value is a UUID4 generated hex string.
    name : str
        The name of the data store.
    data_coordinate_systems : list[DataCoordinateSystem]
        The store's intrinsic (voxel) coordinate systems, one per resolution
        level, index ``0`` being the finest.  A single-resolution store has
        one entry.

        **Stored, not derived.**  Transforms serialize their endpoints as
        UUIDs, so a system rebuilt on load would mint fresh ids and every
        transform naming it would point at nothing.

        The default is an **empty list**: a store with no axis metadata does
        not invent one, because ``Axis.axis_type`` has no honest default.
        Concrete stores that can say -- the OME-Zarr readers -- populate it at
        construction, and the in-memory stores take the ``axis_names`` /
        ``axis_types`` shorthand.  A store that reaches
        ``CellierController.add_visual`` still empty has one derived from the
        scene's world axes; see ``CellierController._ensure_data_coordinate_systems``.
    level_transforms : list[AffineTransform]
        Level ``k`` voxel space -> level ``0`` voxel space, one per entry in
        ``data_coordinate_systems``.  Index ``0`` is the identity.  Empty
        alongside an empty ``data_coordinate_systems``.

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
    data_coordinate_systems: list[DataCoordinateSystem] = Field(default_factory=list)
    level_transforms: list[AffineTransform] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _expand_axis_shorthand(cls, data: Any) -> Any:
        """Turn ``axis_names`` / ``axis_types`` / ``axis_units`` into a system.

        The ergonomic path for a single-level store: naming the axes is one
        keyword rather than a hand-built ``DataCoordinateSystem`` and a UUID.
        Multi-level stores are not covered -- their level transforms carry a
        downsampling factor that no shorthand can supply -- and neither are
        the stores that already declare ``axis_names`` as a field of their
        own (the OME-Zarr readers, which build their systems from NGFF
        metadata in ``from_path``).
        """
        if not isinstance(data, dict) or "axis_names" in cls.model_fields:
            return data
        names = data.pop("axis_names", None)
        types = data.pop("axis_types", None)
        units = data.pop("axis_units", None)
        if names is None:
            if types is not None or units is not None:
                raise ValueError(
                    "axis_types / axis_units need axis_names alongside them; "
                    "there is nothing to attach them to."
                )
            return data
        if data.get("data_coordinate_systems"):
            raise ValueError(
                "Pass either axis_names or data_coordinate_systems, not both."
            )
        store_id = data.get("id")
        if store_id is None:
            store_id = uuid4()
        elif isinstance(store_id, str):
            store_id = uuid.UUID(store_id, version=4)
        data["id"] = store_id
        name = data.get("name") or cls.model_fields["name"].default
        system = data_coordinate_system(store_id, build_axes(names, types, units), name)
        data["data_coordinate_systems"] = [system]
        data["level_transforms"] = [identity_transform(system, system)]
        return data

    @property
    def data_coordinate_system(self) -> DataCoordinateSystem:
        """The level-0 (intrinsic) coordinate system.

        Returns
        -------
        DataCoordinateSystem
            The finest-resolution system.

        Raises
        ------
        ValueError
            If the store has no coordinate systems.  The message names the
            two ways to supply them, because this is the error a caller who
            built a bare in-memory store outside a viewer will hit.
        """
        if not self.data_coordinate_systems:
            raise ValueError(
                f"Data store '{self.name}' has no data_coordinate_systems.  "
                f"Pass axis_names= (and axis_types= for any non-spatial axis) "
                f"when constructing it, or add it to a scene, which derives "
                f"them from the scene's world axes."
            )
        return self.data_coordinate_systems[0]

    def set_data_coordinate_systems(
        self,
        systems: list[DataCoordinateSystem],
        level_transforms: list[AffineTransform] | None = None,
    ) -> None:
        """Install the store's coordinate systems and level transforms.

        Assignment rather than construction because the systems are sometimes
        derived from the scene the store is added to, which the store cannot
        see at construction time.

        Parameters
        ----------
        systems : list[DataCoordinateSystem]
            One per resolution level, finest first.
        level_transforms : list[AffineTransform] or None
            Level ``k`` -> level ``0``.  ``None`` is allowed only for a
            single-level store, where the transform is the identity.

        Raises
        ------
        ValueError
            If *systems* is empty, if *level_transforms* has a different
            length, or if a multi-level store is given no level transforms --
            there is no downsampling factor to infer them from here.
        """
        if not systems:
            raise ValueError(
                "set_data_coordinate_systems needs at least the level-0 system."
            )
        if level_transforms is None:
            if len(systems) > 1:
                raise ValueError(
                    f"A {len(systems)}-level store needs explicit "
                    f"level_transforms; only a single-level store's transform "
                    f"is the identity."
                )
            level_transforms = [identity_transform(systems[0], systems[0])]
        if len(level_transforms) != len(systems):
            raise ValueError(
                f"level_transforms must have one entry per coordinate system; "
                f"got {len(level_transforms)} for {len(systems)} systems."
            )
        self.data_coordinate_systems = list(systems)
        self.level_transforms = list(level_transforms)

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
