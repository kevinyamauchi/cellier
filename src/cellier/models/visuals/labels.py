"""Visual for display label images."""

from typing import Literal

from cellier.models.visuals.base import BaseMaterial, BaseVisual


class LabelsMaterial(BaseMaterial):
    """Material for a labels visual.

    Parameters
    ----------
    color_map : str
        The color map to use for the labels.
    """

    color_map: str


class MultiscaleLabelsVisual(BaseVisual):
    """Model for a multiscale labels visual.

    Parameters
    ----------
    name : str
        The name of the visual
    data_store_id : str
        The id of the data store to be visualized.
    """

    data_store_id: str
    downscale_factors: list[int]
    material: LabelsMaterial

    # this is used for a discriminated union
    visual_type: Literal["labels"] = "labels"
