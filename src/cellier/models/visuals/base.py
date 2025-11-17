"""Base classes for nodes and materials."""

from typing import Any
from uuid import uuid4

import numpy as np
from psygnal import EmissionInfo, EventedModel
from pydantic import Field

from cellier.transform import AffineTransform


def default_transform() -> AffineTransform:
    """Create a default identity transform."""
    return AffineTransform(matrix=np.eye(4))


class BaseAppearance(EventedModel):
    """Base model for all materials.

    Parameters
    ----------
    visible : bool
        If True, the visual is visible.
        Default value is True.
    """

    visible: bool = True


class BaseVisual(EventedModel):
    """Base model for all nodes.

    Parameters
    ----------
    name : str
        The name of the node.
    appearance : BaseAppearance
        The appearance of the visual.
        This should be overridden with the visual-specific
        implementation in the subclasses.
    transform : AffineTransform
        The transform of the visual from data space to world space.
        Default is the identity.
    pick_write : bool
        If True, the visual can be picked.
        Default value is True.
    """

    name: str
    appearance: BaseAppearance
    transform: AffineTransform = Field(default_factory=default_transform)
    pick_write: bool = True

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)

    def model_post_init(self, __context: Any) -> None:
        """Function called after the model is initialized."""
        # connect all appearance events to the visual's appearance signal
        self.appearance.events.all.connect(self._on_appearance_change)

    def update_state(self, new_state):
        """Update the state of the visual.

        This is often used as a callback for when
        the visual controls update.
        """
        # remove the id field from the new state if present
        new_state.pop("id", None)

        # update the visual with the new state
        self.update(new_state)

    def _on_appearance_change(self, event: EmissionInfo):
        """Callback to relay appearance changes.

        This emits the BaseVisual.events.appearance signal.
        """
        property_name = event.signal.name
        property_value = event.args[0]
        self.events.appearance.emit({property_name: property_value})
