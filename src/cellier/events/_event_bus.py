"""Class to manage connecting events between the model and the view."""

import logging

from cellier.events._visual import VisualEventBus

logger = logging.getLogger(__name__)


class EventBus:
    """Class to manage connecting events between the model and the view.

    There are three types of events:
        - visual: communicate changes to the visual model state.
        - visual_controls: communicate changes to the visual gui state.
    """

    def __init__(self):
        # the signals for each visual model
        self._visual_bus = VisualEventBus()

    @property
    def visual(self) -> VisualEventBus:
        """Return the visual events."""
        return self._visual_bus
