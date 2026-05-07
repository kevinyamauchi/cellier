# src/cellier/v2/visuals/_mesh_memory.py
from typing import Annotated, Literal, Union

from pydantic import Field

from cellier.v2.visuals._base_visual import BaseAppearance, BaseVisual


class MeshFlatAppearance(BaseAppearance):
    """Flat (unlit) mesh — maps to MeshBasicMaterial.

    No lights required.  Suitable for false-color meshes, wireframe
    overlays, and rapid inspection.

    Parameters
    ----------
    color : tuple[float, float, float, float]
        Uniform RGBA, used when color_mode is ``"uniform"``.
    color_mode : str
        ``"uniform"``, ``"vertex"``, or ``"face"``.
    wireframe : bool
        Render only edges.  Default False.
    wireframe_thickness : float
        Edge thickness in screen pixels.  Default 1.0.
    opacity : float
        0-1 alpha multiplier.  Default 1.0.
    side : str
        ``"both"``, ``"front"``, or ``"back"``.  Default ``"both"``.
    """

    appearance_type: Literal["flat"] = "flat"
    color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    color_mode: Literal["uniform", "vertex", "face"] = "uniform"
    wireframe: bool = False
    wireframe_thickness: float = 1.0
    side: Literal["both", "front", "back"] = "both"


class MeshPhongAppearance(BaseAppearance):
    """Phong-shaded mesh — maps to MeshPhongMaterial.

    Requires lights in the scene.  Pass ``lighting="default"`` to
    ``controller.add_scene()`` to add ambient + directional lights.

    Parameters
    ----------
    color : tuple[float, float, float, float]
        Uniform RGBA diffuse color, used when color_mode is
        ``"uniform"``.
    color_mode : str
        ``"uniform"``, ``"vertex"``, or ``"face"``.
    shininess : float
        Specular exponent.  Default 30.
    opacity : float
        0-1 alpha multiplier.  Default 1.0.
    side : str
        ``"both"``, ``"front"``, or ``"back"``.  Default ``"front"``.
    flat_shading : bool
        Use face normals instead of smooth vertex normals.
        Default False.
    """

    appearance_type: Literal["phong"] = "phong"
    color: tuple[float, float, float, float] = (0.4, 0.6, 0.9, 1.0)
    color_mode: Literal["uniform", "vertex", "face"] = "uniform"
    shininess: float = 30.0
    side: Literal["both", "front", "back"] = "front"
    flat_shading: bool = False


MeshAppearance = Annotated[
    Union[MeshFlatAppearance, MeshPhongAppearance],
    Field(discriminator="appearance_type"),
]


class MeshVisual(BaseVisual):
    """Model-layer visual for an in-memory triangle mesh.

    Parameters
    ----------
    visual_type : Literal["mesh_memory"]
        Discriminator field; always ``"mesh_memory"``.
    data_store_id : str
        UUID string of the associated ``MeshMemoryStore``.
    appearance : MeshFlatAppearance | MeshPhongAppearance
        Appearance parameters.
    requires_camera_reslice : bool
        Always False; frozen.  Camera movement does not trigger reslicing.
    """

    visual_type: Literal["mesh_memory"] = "mesh_memory"
    data_store_id: str
    appearance: MeshAppearance
    requires_camera_reslice: bool = Field(default=False, frozen=True)
