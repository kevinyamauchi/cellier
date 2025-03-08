"""Models for Lines nodes."""

from typing import Literal

from cellier.models.visuals.base import BaseMaterial, BaseVisual


class LinesUniformMaterial(BaseMaterial):
    """Give all lines the same appearance.

    Parameters
    ----------
    size : float
        The size of the points in the units
        specified by size_coordinate_space.
    color : tuple[float, float, float, float]
        RGBA color for all of the points.
    opacity : float
        The opacity of the lines from 0 to 1 where 1 is opaque.
        Default value is 1.0
    size_coordinate_space : str
        The coordinate space the size is defined in.
        Options are "screen", "world", "data".
        Default value is "data"
    """

    size: float
    color: tuple[float, float, float, float]
    opacity: float = 1.0
    size_coordinate_space: Literal["screen", "world", "data"] = "data"


class LinesVisual(BaseVisual):
    """Model for a lines visual.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    todo: add more lines materials
    todo: decide if different line data types supported.

    Parameters
    ----------
    name : str
        The name of the visual
    data_store_id : str
        The id of the data store to be visualized.
    material : LinesUniformMaterial
        The model for the appearance of the rendered lines.
    pick_write : bool
        If True, the visual can be picked.
        Default value is True.
    id : str
        The unique id of the visual.
        The default value is a uuid4-generated hex string.
    """

    data_store_id: str
    material: LinesUniformMaterial

    # this is used for a discriminated union
    visual_type: Literal["lines"] = "lines"
