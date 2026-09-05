"""Tuning constants shared by both GUI backends.

Values that describe how a control *behaves* rather than what it drives, and
that both front ends have to agree on because a user moving between a script
and a notebook should not find the same control responding differently.

Each was previously written twice -- once per toolkit, and in one case once in
Python and once in JavaScript, where no test could ever have compared them.
"""

from __future__ import annotations

DIMS_SLIDER_THROTTLE_MS: int = 50
"""How often a dims-slider drag is allowed to reach the slicer, in ms.

Dragging a slider produces far more values than the slicer can serve, so both
front ends coalesce them: Qt with a ``QTimer`` in ``QtDimsControl``, anywidget
with a leading-plus-trailing throttle in ``dims_panel.js``.  Two independent
implementations are fine -- the toolkits offer different primitives -- but two
independent *intervals* are not, since the interval is what a user actually
feels.

The anywidget side receives this as a synced trait rather than hard-coding it,
so the JavaScript reads the same number Python does.
"""
