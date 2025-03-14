from enum import Enum

from cellier.models.data_stores import ImageMemoryStore
from cellier.models.visuals import MultiscaleLabelsVisual


class LabelsPaintingMode(Enum):
    """Enum for the different modes of painting labels.

    Attributes
    ----------
    NONE : str
        No painting.
    PAINT : str
        Paint the labels.
    ERASE : str
        Erase the labels.
    """

    NONE = "none"
    PAINT = "paint"
    ERASE = "erase"


class LabelsPaintingManager:
    """Class to manage the painting of labels data.

    Parameters
    ----------
    model : MultiscaleLabelsVisual
        The model for the labels visual to be painted.
        Currently, only labels with a single scale are supported.
    data_store : ImageMemoryStore
        The data store for the labels visual to be painted.
    """

    def __init__(
        self,
        model: MultiscaleLabelsVisual,
        data_store: ImageMemoryStore,
        mode: LabelsPaintingMode = LabelsPaintingMode.NONE,
    ):
        if len(MultiscaleLabelsVisual.downscale_factors) != 1:
            raise NotImplementedError("Only single scale labels are supported.")

        self._model = model
        self._data = data_store
        self._mode = mode
