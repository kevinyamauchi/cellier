"""Visual for display label images."""

from typing import Literal

from cellier.models.visuals.base import BaseAppearance, BaseVisual


class LabelsAppearance(BaseAppearance):
    """Material for a labels visual.

    Parameters
    ----------
    color_map : str
        The color map to use for the labels.
    visible : bool
        If True, the visual is visible.
        Default value is True.
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
    downscale_factors : list[int]
        The downscale factors for each scale level of the labels.
    appearance : LabelsAppearance
        The material to use for the labels visual.
    pick_write : bool
        If True, the visual can be picked.
        Default value is True.
    id : str
        The unique id of the visual.
        The default value is a uuid4-generated hex string.
        Do not populate this field manually.
    """

    data_store_id: str
    downscale_factors: list[int]
    appearance: LabelsAppearance

    # this is used for a discriminated union
    visual_type: Literal["labels"] = "labels"
