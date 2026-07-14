"""Toolkit-neutral colormap helpers shared by the Qt and anywidget GUIs."""

from __future__ import annotations


def colormap_to_str(cm) -> str:
    """Convert a ``cmap.Colormap`` (or string) to its canonical name string.

    Bus events carry the model-native ``cmap.Colormap`` object, while both GUI
    toolkits drive their colormap controls with a name string, so inbound
    handlers normalise through this helper.
    """
    if isinstance(cm, str):
        return cm
    name = getattr(cm, "name", None)
    if name is not None and isinstance(name, str):
        return name
    return str(cm)
