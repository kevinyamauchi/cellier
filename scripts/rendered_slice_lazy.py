# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einops",
#     "glfw",
#     "numpy",
#     "pygfx",
#     "scipy",
# ]
# ///
"""Attempt at lazy slice rendering with an Image node."""

import einops
import numpy as np
import pygfx as gfx
from pylinalg import vec_transform
from rendercanvas.auto import RenderCanvas, loop
from scipy.ndimage import map_coordinates

edge_length = 50
half_length = edge_length // 2

grid_shape = (edge_length, edge_length)
grid_spacing = (
    1,
    1,
)


class LazyImageSlice(gfx.Image):
    def _update_object(self):
        local_last_modified = self.local.last_modified
        if local_last_modified > self._world_last_modified:
            # if the transform has been updated, reslice the object
            print("Updating object")
            transform_matrix = self.world.matrix

            matrix = transform_matrix[:3, :3]
            matrix_flipped = matrix[::-1, ::-1]
            translation = transform_matrix[:3, 3]
            translation_flipped = translation[::-1]

            print(translation_flipped)

            # reshape the sampling grid to be a list of coordinates
            grid_coords = sampling_grid.reshape(-1, 3)

            # apply the transform to the grid
            new_transform = np.eye(4, dtype=np.float32)
            new_transform[:3, :3] = matrix_flipped
            new_transform[:3, 3] = translation_flipped
            transformed_grid = vec_transform(grid_coords, new_transform)

            # apply the translation
            print(f"    min_corner: {transformed_grid.min(axis=0)}")
            print(f"    max_corner: {transformed_grid.max(axis=0)}")

            sampled_volume = map_coordinates(
                image,
                transformed_grid.reshape(-1, 3).T,
                order=0,
                cval=0,
            )

            print(f"    labels: {np.unique(sampled_volume)}")

            self.geometry.grid.data[:] = sampled_volume.reshape(grid_shape)
            self.geometry.grid.update_full()

        super()._update_object()


def generate_3d_grid(
    grid_shape: tuple[int, int, int] = (10, 10, 10),
    grid_spacing: tuple[float, float, float] = (1, 1, 1),
) -> np.ndarray:
    """
    Generate a 3D sampling grid with specified shape and spacing.

    The grid generated is centered on the origin, has shape (w, h, d, 3) for
    grid_shape (w, h, d), and spacing grid_spacing between neighboring points.

    Parameters
    ----------
    grid_shape : Tuple[int, int, int]
        The number of grid points along each axis.
    grid_spacing : Tuple[float, float, float]
        Spacing between points in the sampling grid.

    Returns
    -------
    np.ndarray
        Coordinate of points forming the 3D grid.
    """
    # generate a grid of points at each integer from 0 to grid_shape for each dimension
    grid = np.indices(grid_shape).astype(float)
    grid = einops.rearrange(grid, "xyz w h d -> w h d xyz")
    # shift the grid to be centered on the origin
    # grid_offset = (np.array(grid_shape)) // 2
    # grid -= grid_offset
    # scale the grid to get correct spacing
    grid *= grid_spacing
    return grid


def generate_2d_grid(
    grid_shape: tuple[int, int] = (10, 10), grid_spacing: tuple[float, float] = (1, 1)
) -> np.ndarray:
    """
    Generate a 2D sampling grid with specified shape and spacing.

    The grid generated is centered on the origin, lying on the plane with normal
    vector [1, 0, 0], has shape (w, h, 3) for grid_shape (w, h), and spacing
    grid_spacing between neighboring points.

    Parameters
    ----------
    grid_shape : Tuple[int, int]
        The number of grid points along each axis.
    grid_spacing : Tuple[float, float]
        Spacing between points in the sampling grid.

    Returns
    -------
    np.ndarray
        Coordinate of points forming the 2D grid.
    """
    grid = generate_3d_grid(
        grid_shape=(1, *grid_shape), grid_spacing=(1, *grid_spacing)
    )
    return einops.rearrange(grid, "1 w h xyz -> w h xyz")


def create_3d_cube_quadrants(size=8):
    r"""
    Create a 3D NumPy array representing a cube where each octant
    has a unique value from 0 to 7, using ZYX indexing.

       5-------6
      /|      /|
     /  |    / |
    7-------8  |
    |  1---|--2  |
    | /    | /   |
    3-------4    /
     \           /
       \       /
         Bottom

    Octant Values:
    1: Bottom Back Left   (0, 0, 0)
    2: Bottom Back Right  (0, 1, 0)
    3: Bottom Front Left  (0, 0, 1)
    4: Bottom Front Right (0, 1, 1)
    5: Top Back Left      (1, 0, 0)
    6: Top Back Right     (1, 1, 0)
    7: Top Front Left     (1, 0, 1)
    8: Top Front Right    (1, 1, 1)

    Parameters
    ----------
    size : int, optional (default=8)
        The edge length of each dimension of the cube

    Returns
    -------
    np.ndarray
        A 3D array with 8 distinct octant values
    np.ndarray
        (8,3) array of centroids for each octant
    """
    # Create an empty cube
    cube = np.zeros((size, size, size), dtype=np.float32)

    # Calculate the midpoint
    mid = size // 2

    # Assign values to each octant using direct indexing (ZYX order)
    cube[0:mid, 0:mid, 0:mid] = 1  # Bottom Back Left
    cube[0:mid, mid:, 0:mid] = 2  # Bottom Back Right
    cube[0:mid, 0:mid, mid:] = 3  # Bottom Front Left
    cube[0:mid, mid:, mid:] = 4  # Bottom Front Right
    cube[mid:, 0:mid, 0:mid] = 5  # Top Back Left
    cube[mid:, mid:, 0:mid] = 6  # Top Back Right
    cube[mid:, 0:mid, mid:] = 7  # Top Front Left
    cube[mid:, mid:, mid:] = 8  # Top Front Right

    # normalize the cube to be in the range [0, 1]
    cube /= 9

    # get the centroids of each octant
    centroids = np.array(
        [
            [mid / 2, mid / 2, mid / 2],  # 0: Bottom Back Left
            [mid / 2, mid * 3 / 2, mid / 2],  # 1: Bottom Back Right
            [mid / 2, mid / 2, mid * 3 / 2],  # 2: Bottom Front Left
            [mid / 2, mid * 3 / 2, mid * 3 / 2],  # 3: Bottom Front Right
            [mid * 3 / 2, mid / 2, mid / 2],  # 4: Top Back Left
            [mid * 3 / 2, mid * 3 / 2, mid / 2],  # 5: Top Back Right
            [mid * 3 / 2, mid / 2, mid * 3 / 2],  # 6: Top Front Left
            [mid * 3 / 2, mid * 3 / 2, mid * 3 / 2],  # 7: Top Front Right
        ],
        dtype=np.float32,
    )

    return cube, centroids


# make the image
image, coordinates = create_3d_cube_quadrants(size=edge_length)

# color map
color_names = [
    "Black",
    "White",
    "Blue",
    "Magenta",
    "Green",
    "Cyan",
    "Orange",
    "Gray",
    "Yellow",
]
colors = np.array(
    [
        [0.0, 0.0, 0.0],  # Background: Black
        [1.0, 1.0, 1.0],  # Octant 0: White
        [0.0, 0.0, 1.0],  # Octant 1: Blue (standard blue)
        [1.0, 0.0, 1.0],  # Octant 2: Magenta (pure magenta)
        [0.0, 1.0, 0.0],  # Octant 3: Green (pure green)
        [0.0, 1.0, 1.0],  # Octant 4: Cyan (pure cyan)
        [1.0, 0.5, 0.0],  # Octant 5: Orange (standard orange)
        [0.5, 0.5, 0.5],  # Octant 6: Gray (medium gray)
        [1.0, 1.0, 0.0],  # Octant 7: Yellow (pure yellow)
    ],
    dtype=np.float32,
)
colormap = gfx.TextureMap(
    texture=gfx.Texture(colors, dim=1), filter="nearest", wrap="clamp"
)

# make the scene/canvas
canvas = RenderCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
viewport = gfx.Viewport(renderer)
scene = gfx.Scene()

# add the axes
scene.add(gfx.AxesHelper(size=40, thickness=5))

# make the volume node
geometry = gfx.Geometry(grid=image)
material = gfx.VolumeIsoMaterial(clim=(0, 1), threshold=0, map=colormap)
volume_node = gfx.Volume(geometry, material, visible=False)
scene.add(volume_node)

bounding_box = gfx.BoxHelper(color="red")
bounding_box.set_transform_by_object(volume_node)
scene.add(bounding_box)

# make the image node
tex = gfx.Texture(np.zeros_like(image[0, ...]), dim=2)
image_node = LazyImageSlice(
    gfx.Geometry(grid=tex),
    gfx.ImageBasicMaterial(clim=(0, 1), interpolation="nearest", map=colormap),
)
scene.add(image_node)

# make the points node
points_geometry = gfx.Geometry(
    positions=coordinates[:, [2, 1, 0]],  # ZYX order
    colors=colors[1:],
)
points_material = gfx.PointsMaterial(size=5, color_mode="vertex")
points_node = gfx.Points(points_geometry, points_material)
scene.add(points_node)

# make the camera
camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.show_object(scene, view_dir=(-1, -1, -1), up=(0, 0, 1))
controller = gfx.OrbitController(camera, register_events=renderer)


# add the gizmo
gizmo = gfx.TransformGizmo(image_node)
gizmo.add_default_event_handlers(viewport, camera)

sampling_grid = generate_2d_grid(grid_shape=grid_shape, grid_spacing=grid_spacing)


def on_key_press(event):
    if event.key == "q":
        # toggle the visibility of the volume
        if volume_node.visible:
            volume_node.visible = False
        else:
            volume_node.visible = True


def animate():
    viewport.render(scene, camera)
    viewport.render(gizmo, camera)
    renderer.flush()


renderer.add_event_handler(on_key_press, "key_down")

print("Point coordinates:")
for octant_index, (coordinate, color) in enumerate(zip(coordinates, color_names[1:])):
    print(f"    Octant {octant_index + 1} coordinates: {coordinate}, color: {color}")


if __name__ == "__main__":
    canvas.request_draw(animate)
    loop.run()
