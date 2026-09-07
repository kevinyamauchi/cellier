"""Qt panel for the temporal accumulation settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from cellier.gui._render_controls import RENDER_SECTION_TITLES
from cellier.gui.qt.render._base import QtRenderConfigPanel

if TYPE_CHECKING:
    from collections.abc import Callable

    from cellier.render._config import TemporalAccumulationConfig

#: Frames after which the exponential average has essentially converged at
#: the default alpha of 0.1.  Only used to word the readout.
_SETTLED_FRAMES = 10


class QtTemporalControls(QtRenderConfigPanel):
    """Temporal accumulation settings on the cellier bus.

    The pass blends successive frames with an exponential moving average.
    Because the volume raymarcher and the occlusion kernel both jitter per
    frame, that average is what lets them use few samples per frame and
    still settle to a clean image once the camera stops -- it is why the
    occlusion sample count can default to 16 rather than 64.

    This is the one feature whose entire behaviour is invisible unless you
    know to hold the camera still and count, so the panel carries a
    convergence readout as well as its two settings.  The pass is off in 2D
    whatever ``enabled`` says.

    Parameters
    ----------
    config :
        The live ``TemporalAccumulationConfig`` to read initial values
        from -- typically ``controller.render_config.temporal``.
    frame_count :
        Called to read how many frames have accumulated on the
        least-converged canvas, for the readout.  Typically
        ``lambda: controller.render_manager.temporal_frame_count``.
        ``None`` omits the readout.
    on_reset :
        Called when the user asks to discard the accumulated history.
        Typically ``controller.reset_temporal_accumulation``.  ``None``
        omits the button.  This is an action rather than a setting, so it
        does not travel on the bus.
    parent :
        Optional Qt parent widget.
    """

    section: ClassVar[str] = "temporal"
    title: ClassVar[str] = RENDER_SECTION_TITLES["temporal"]

    def __init__(
        self,
        config: TemporalAccumulationConfig,
        *,
        frame_count: Callable[[], int | None] | None = None,
        on_reset: Callable[[], None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._frame_count = frame_count

        self._build_from_spec(config)

        if frame_count is not None:
            self._add_readout("State", self._describe_convergence)

        if on_reset is not None:
            from qtpy.QtWidgets import QPushButton

            button = QPushButton("Reset history", self._container)
            button.setToolTip(
                "Discard the accumulated average and start again from the "
                "next frame.  Cellier already does this on every camera and "
                "content change."
            )
            button.clicked.connect(lambda: (on_reset(), self.refresh_readouts()))
            self._layout.addWidget(button)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _describe_convergence(self) -> str:
        """Say whether the image has settled, in words rather than a number."""
        count = self._frame_count() if self._frame_count is not None else None
        if count is None:
            return "no canvas yet"
        if count == 0:
            return "restarting"
        if count < _SETTLED_FRAMES:
            return f"settling ({count} frames)"
        return f"settled ({count} frames)"
