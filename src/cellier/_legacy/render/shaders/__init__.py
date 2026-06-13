"""Module of custom pygfx shaders."""

from cellier._legacy.render.shaders.labels import (
    LabelImageMaterial,
    LabelImageShader,
    LabelIsoMaterial,
    LabelIsoShader,
)

__all__ = [
    "LabelIsoShader",
    "LabelIsoMaterial",
    "LabelImageShader",
    "LabelImageMaterial",
]
