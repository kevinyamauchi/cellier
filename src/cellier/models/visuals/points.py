"""Models for Points nodes."""

from typing import Literal

from cellier.models.visuals.base import BaseMaterial, BaseVisual


class PointsUniformMaterial(BaseMaterial):
    """Give all points the same appearance.

    Parameters
    ----------
    size : float
        The size of the points in the units
        specified by size_coordinate_space.
    color : Tuple[float, float, float, float]
        RGBA color for all of the points.
    size_coordinate_space : str
        The coordinate space the size is defined in.
        Options are "screen", "world", "data".
        Default value is "data"
    """

    size: float
    color: tuple[float, float, float, float]
    size_coordinate_space: Literal["screen", "world", "data"] = "data"


class PointsVisual(BaseVisual):
    """Model for a point cloud visual.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    todo: add more point materials

    Parameters
    ----------
    name : str
        The name of the visual
    data_store_id : str
        The id of the data stream to be visualized.
    material : PointsUniformMaterial
        The model for the appearance of the rendered points.
    pick_write : bool
        If True, the visual can be picked.
        Default value is True.
    id : str
        The unique id of the visual.
        The default value is a uuid4-generated hex string.
    """

    data_store_id: str
    material: PointsUniformMaterial

    # this is used for a discriminated union
    visual_type: Literal["points"] = "points"
