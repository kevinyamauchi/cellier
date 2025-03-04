"""Base classes for nodes and materials."""

from uuid import uuid4

from psygnal import EventedModel
from pydantic import Field


class BaseVisual(EventedModel):
    """Base model for all nodes.

    Parameters
    ----------
    name : str
        The name of the node.
    pick_write : bool
        If True, the node can be picked.
        Default value is True.
    """

    name: str
    pick_write: bool = True

    # store a UUID to identify this specific scene.
    id: str = Field(default_factory=lambda: uuid4().hex)

    def update_state(self, new_state):
        """Update the state of the visual.

        This is often used as a callback for when
        the visual controls update.
        """
        # remove the id field from the new state if present
        new_state.pop("id", None)

        # update the visual with the new state
        self.update(new_state)


class BaseMaterial(EventedModel):
    """Base model for all materials."""

    pass
