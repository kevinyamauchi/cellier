"""Host-aware placement for the anywidget front-end (the ``LayoutHost`` seam).

The leaves are host-uniform (the per-control appearance anywidgets, the
``rendercanvas.anywidget`` canvas, the toggle).  Only composition and
presentation differ between Jupyter and marimo, isolated behind one injected
object (design doc section 10).

``gui="anywidget"`` selects the toolkit; the *host* (Jupyter vs marimo) is
detected by default with an explicit ``host=`` override via
:func:`resolve_host`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


@runtime_checkable
class LayoutHost(Protocol):
    """Composition + presentation seam for one anywidget host."""

    def leaf(self, widget: object) -> object:
        """Wrap one anywidget leaf for this host."""
        ...

    def stack(
        self,
        items: Sequence[object],
        *,
        direction: str = "v",
        align: str | None = None,
        min_width: int | None = None,
        gap: int | None = None,
    ) -> object:
        """Stack *items* vertically (``"v"``) or horizontally (``"h"``).

        *align* sets the cross-axis alignment of the items (e.g. ``"center"``
        to centre a fixed-width canvas over a wider control panel); ``None``
        leaves the host default.

        *min_width* makes the stack grow to fill available width but never
        narrower than *min_width* pixels; ``None`` leaves the host default
        (content-sized, no grow).

        *gap* sets the spacing between items in pixels; ``None`` leaves the
        host default (tuned for macro layout blocks like canvas/dims/docks).
        Pass a small explicit value to tightly group sibling controls that
        used to live inside one widget (see
        :func:`cellier.convenience.gui._appearance_widgets.compose_appearance_leaf`).
        """
        ...

    def grid(self, rows: Sequence[Sequence[object]]) -> object:
        """Arrange *rows* (a list of rows of items) as a grid."""
        ...

    def present(self, root: object) -> object | None:
        """Render *root*, or return it for ``display()`` to yield.

        Two host conventions, distinguished by the return value:

        * Imperative hosts (Jupyter) render *root* as a side effect (e.g.
          ``IPython.display.display``) and return ``None``; ``display()`` then
          returns an inert handle so the cell shows a single copy.
        * Return-value hosts (marimo) render the cell's *returned* value, so
          ``present`` returns *root* and ``display()`` yields it as the cell
          output (an imperative ``mo.output.replace`` is overridden by the
          cell's last expression, so it cannot be used here).
        """
        ...


class MarimoHost:
    """marimo host -- native anywidget + layout primitives."""

    def __init__(self) -> None:
        import marimo as mo

        self._mo = mo

    def leaf(self, widget: object) -> object:
        """Wrap a leaf with ``marimo.ui.anywidget``."""
        return self._mo.ui.anywidget(widget)

    # marimo's own vstack/hstack ``gap=`` is in rem, not pixels; this converts
    # our pixel-based LayoutHost.stack(gap=...) contract to marimo's unit
    # assuming the standard 16px-per-rem base.
    _REM_PX = 16

    def stack(
        self,
        items: Sequence[object],
        *,
        direction: str = "v",
        align: str | None = None,
        min_width: int | None = None,
        gap: int | None = None,
    ) -> object:
        """Stack with ``marimo.vstack`` / ``marimo.hstack``.

        *min_width* is accepted for interface parity with :class:`JupyterHost`
        but ignored -- marimo has its own layout/width primitives.

        *gap* (pixels, matching :class:`JupyterHost`) is converted to
        marimo's own rem-based ``gap=`` only when given; omitted, marimo's
        own default is used unchanged.  ``mo.vstack``/``mo.hstack`` default to
        a non-trivial gap meant for spacing distinct layout blocks apart,
        which is too loose for grouping sibling controls that used to live
        inside one widget.
        """
        stacker = self._mo.vstack if direction == "v" else self._mo.hstack
        kwargs = {} if gap is None else {"gap": gap / self._REM_PX}
        return stacker(list(items), align=align, **kwargs)

    def grid(self, rows: Sequence[Sequence[object]]) -> object:
        """Arrange rows with nested ``vstack`` / ``hstack``."""
        return self._mo.vstack([self._mo.hstack(list(r)) for r in rows])

    def present(self, root: object) -> object | None:
        """Return *root* so ``display()`` yields it as the cell output.

        marimo renders a cell's *last expression*, which overrides an
        imperative ``mo.output.replace``; so instead of rendering here, we hand
        *root* back and let ``display()`` return it.
        """
        return root


class JupyterHost:
    """Jupyter host -- manager-rendered anywidget container (``AnywidgetBox``)."""

    def leaf(self, widget: object) -> object:
        """A ``DOMWidget`` is directly displayable; pass it through."""
        return widget

    def stack(
        self,
        items: Sequence[object],
        *,
        direction: str = "v",
        align: str | None = None,
        min_width: int | None = None,
        gap: int | None = None,
    ) -> object:
        """Compose into an ``AnywidgetBox`` flexbox."""
        from cellier.gui.anywidget import AnywidgetBox

        kwargs = {} if gap is None else {"gap": gap}
        return AnywidgetBox(
            children=list(items),
            direction=direction,
            align=align or "",
            min_width=min_width or 0,
            **kwargs,
        )

    def grid(self, rows: Sequence[Sequence[object]]) -> object:
        """Compose rows of ``AnywidgetBox`` (horizontal) inside an outer one."""
        from cellier.gui.anywidget import AnywidgetBox

        return AnywidgetBox(
            children=[AnywidgetBox(children=list(r), direction="h") for r in rows],
            direction="v",
        )

    def present(self, root: object) -> object | None:
        """Render *root* imperatively via ``IPython.display.display``.

        Returns ``None`` so ``display()`` yields an inert handle (the viewer is
        already shown), avoiding a duplicate copy from the cell's return value.
        """
        from IPython.display import display as ipy_display

        ipy_display(root)
        return None


def _marimo_running() -> bool:
    try:
        import marimo as mo

        return bool(mo.running_in_notebook())
    except Exception:
        return False


def _ipython_running() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def resolve_host(host: str | None = None) -> LayoutHost:
    """Resolve the anywidget :class:`LayoutHost`.

    The explicit *host* override wins; otherwise the host is detected (marimo
    first, then IPython / Jupyter).

    Parameters
    ----------
    host : "marimo", "jupyter", or None
        Explicit host override, or ``None`` to auto-detect.

    Returns
    -------
    LayoutHost

    Raises
    ------
    RuntimeError
        If no host is given and none can be detected.
    """
    if host == "marimo" or (host is None and _marimo_running()):
        return MarimoHost()
    if host == "jupyter" or (host is None and _ipython_running()):
        return JupyterHost()
    raise RuntimeError(
        "No anywidget host detected; pass host='jupyter' or host='marimo'."
    )
