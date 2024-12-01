"""Visual and material models for meshes."""

from typing import Literal, Tuple, Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.models.nodes.base_node import BaseMaterial, BaseNode


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

    # this is used for a discriminated union
    material_type: Literal["phong"] = "phong"


class MeshStandardMaterial(BaseMaterial):
    """Physically-based rendering shaded mesh.

    See the PyGFX MeshStandardMaterial.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    emissive : Tuple[float, float, float]
        The emissive (light) color of the mesh.
        This color is added to the final color and is unaffected by lighting.
    metalness : float
        How much the material looks like metal. 0.0 is no metal, and 1.0 is metal.
    roughness : float
        How rough the mesh appears. 0.0 is smooth and 1.0 is rough.
    """

    emissive: Tuple[float, float, float] = (0, 0, 0)
    metalness: float = 0
    roughness: float = 1.0

    # this is used for a discriminated union
    material_type: Literal["standard"] = "standard"


MeshMaterialType = Annotated[
    Union[MeshPhongMaterial, MeshStandardMaterial], Field(discriminator="material_type")
]


class MeshNode(BaseNode):
    """Model for a mesh visual.

    This is a psygnal EventedModel.
    https://psygnal.readthedocs.io/en/latest/API/model/

    Parameters
    ----------
    name : str
        The name of the visual
    data_stream_id : str
        The id of the data stream to be visualized.
    material : BaseMeshMaterial
        The model for the appearance of the rendered mesh.
    """

    data_stream_id: str
    material: MeshMaterialType

    # this is used for a discriminated union
    visual_type: Literal["mesh"] = "mesh"
