"""Visual and material models for meshes."""

from typing import Tuple

from cellier.models.data_streams.mesh import BaseMeshDataStream
from cellier.models.visuals.base_visual import BaseMaterial, BaseVisual


class BaseMeshMaterial(BaseMaterial):
    """Base model for all mesh materials."""

    pass


class MeshPhongMaterial(BaseMeshMaterial):
    """Phong mesh shading model.

    https://en.wikipedia.org/wiki/Phong_shading

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    shininess : int
        How shiny the specular highlight is; a higher value gives a sharper highlight.
    emissive : Tuple[float, float, float]
        The emissive (light) color of the mesh.
        This color is added to the final color and is unaffected by lighting.
    specular : Tuple[float, float, float]
        The highlight color of the mesh.
    """

    shininess: int = 30
    emissive: Tuple[float, float, float] = (0, 0, 0)
    specular: Tuple[float, float, float] = (0.28, 0.28, 0.28)


class MeshVisual(BaseVisual):
    """Model for a mesh visual.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    name : str
        The name of the visual
    data_stream : BaseMeshDataStream
        The data to be visualized.
    material : BaseMeshMaterial
        The model for the appearance of the rendered mesh.
    """

    data_stream: BaseMeshDataStream
    material: BaseMeshMaterial
