"""Models for Lines nodes."""

from typing import Literal, Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.models.visuals.base import BaseAppearance, BaseVisual


class LinesUniformAppearance(BaseAppearance):
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
    visible : bool
        If True, the visual is visible.
        Default value is True.
    """

    size: float
    color: tuple[float, float, float, float]
    opacity: float = 1.0
    size_coordinate_space: Literal["screen", "world", "data"] = "data"

    # this is used for a discriminated union
    appearance_type: Literal["uniform"] = "uniform"


class LinesVertexColorAppearance(BaseAppearance):
    """Color each line with a vertex-specific color.

    Parameters
    ----------
    size : float
        The size of the points in the units
        specified by size_coordinate_space.
    opacity : float
        The opacity of the lines from 0 to 1 where 1 is opaque.
        Default value is 1.0
    size_coordinate_space : str
        The coordinate space the size is defined in.
        Options are "screen", "world", "data".
        Default value is "data"
    visible : bool
        If True, the visual is visible.
        Default value is True.
    """

    size: float
    opacity: float = 1.0
    size_coordinate_space: Literal["screen", "world", "data"] = "data"

    # this is used for a discriminated union
    appearance_type: Literal["vertex_color"] = "vertex_color"


# This is used for a discriminated union for typing the visual models
LinesAppearance = Annotated[
    Union[LinesUniformAppearance, LinesVertexColorAppearance],
    Field(discriminator="appearance_type"),
]


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
    appearance : LinesAppearance
        The model for the appearance of the rendered lines.
    pick_write : bool
        If True, the visual can be picked.
        Default value is True.
    id : str
        The unique id of the visual.
        The default value is a uuid4-generated hex string.
        Do not populate this field manually.
    """

    data_store_id: str
    appearance: LinesAppearance

    # this is used for a discriminated union
    visual_type: Literal["lines"] = "lines"
