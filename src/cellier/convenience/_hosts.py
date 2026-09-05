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

from contextlib import suppress
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
        title: str | None = None,
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

        *title* draws a heading above the items, naming what the stack holds;
        ``None`` draws none.  The anywidget answer to the Qt renderer's
        ``titled_group``, so a dock can say whose settings it carries on
        either toolkit.
        """
        ...

    def grid(self, rows: Sequence[Sequence[object]]) -> object:
        """Arrange *rows* (a list of rows of items) as a grid.

        A ``None`` in a row is an empty cell and must hold its column: the
        items after it keep their position rather than shifting left.  Both
        hosts substitute an invisible placeholder, which is the cheapest way
        to say "nothing here" in a flexbox row.
        """
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
        title: str | None = None,
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
        stacked = stacker(list(items), align=align, **kwargs)
        if not title:
            return stacked
        # marimo has no titled container, so the heading is markdown above the
        # stack -- the same two elements the Jupyter box draws, composed with
        # marimo's own primitives.
        return self._mo.vstack([self._mo.md(f"**{title}**"), stacked])

    def grid(self, rows: Sequence[Sequence[object]]) -> object:
        """Arrange rows with nested ``vstack`` / ``hstack``.

        An empty cell becomes empty markdown, which occupies its column
        without drawing anything.
        """
        return self._mo.vstack(
            [
                self._mo.hstack(
                    [self._mo.md("") if item is None else item for item in row]
                )
                for row in rows
            ]
        )

    def present(self, root: object) -> object | None:
        """Return *root* so ``display()`` yields it as the cell output.

        marimo renders a cell's *last expression*, which overrides an
        imperative ``mo.output.replace``; so instead of rendering here, we hand
        *root* back and let ``display()`` return it.
        """
        return root


class JupyterHost:
    """Jupyter host -- manager-rendered anywidget container (``AnywidgetBox``)."""

    def __init__(self) -> None:
        # Widgets rendered by ``present``; see ``close_presented``.
        self._presented: list[object] = []

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
        title: str | None = None,
    ) -> object:
        """Compose into an ``AnywidgetBox`` flexbox."""
        from cellier.gui.anywidget import AnywidgetBox

        kwargs = {} if gap is None else {"gap": gap}
        return AnywidgetBox(
            children=list(items),
            direction=direction,
            align=align or "",
            min_width=min_width or 0,
            title=title or "",
            **kwargs,
        )

    def grid(self, rows: Sequence[Sequence[object]]) -> object:
        """Compose rows of ``AnywidgetBox`` (horizontal) inside an outer one.

        An empty cell becomes an empty ``AnywidgetBox``: ``children`` only
        accepts ``DOMWidget``s, so the hole has to be a widget, and an
        childless box renders as nothing while still taking its place in the
        row.
        """
        from cellier.gui.anywidget import AnywidgetBox

        return AnywidgetBox(
            children=[
                AnywidgetBox(
                    children=[AnywidgetBox() if item is None else item for item in row],
                    direction="h",
                )
                for row in rows
            ],
            direction="v",
        )

    #: Outer padding (pixels) so the composed tree doesn't touch the notebook
    #: cell / sidecar tab edges; applies to both ``sidecar=True`` and
    #: ``sidecar=False`` since both flow through this one ``present()``.
    _OUTER_PADDING = 12

    def present(self, root: object) -> object | None:
        """Render *root* imperatively via ``IPython.display.display``.

        Wraps *root* in an outer ``AnywidgetBox`` with a small padding so the
        canvas/docks don't touch the cell (or sidecar tab) boundary; nested
        boxes built by cellier.convenience.layout._anywidget_renderer.render_anywidget
        keep ``padding=0``, so this is the only border added.

        Returns ``None`` so ``display()`` yields an inert handle (the viewer is
        already shown), avoiding a duplicate copy from the cell's return value.
        """
        from IPython.display import display as ipy_display

        from cellier.gui.anywidget import AnywidgetBox

        wrapped = AnywidgetBox(children=[root], padding=self._OUTER_PADDING)
        # Kept so it can be closed later.  Because this host renders as a side
        # effect and returns ``None``, the caller never sees this wrapper, so
        # without a reference here nothing could ever release it -- and it is
        # an ``ipywidgets`` widget like any other (see
        # ``cellier.gui.anywidget._teardown``).
        self._presented.append(wrapped)
        ipy_display(wrapped)
        return None

    def close_presented(self) -> None:
        """Close everything this host has rendered.

        Optional part of the ``LayoutHost`` contract, called by
        ``DisplayHandle.close``.  A host that hands its root back to the caller
        (``MarimoHost``) does not need it: whoever owns the root closes it.
        """
        for widget in self._presented:
            close = getattr(widget, "close", None)
            if close is not None:
                with suppress(Exception):
                    close()
        self._presented.clear()


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
