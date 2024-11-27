"""Example showing progressive loading of volume data."""

import numpy as np
import pygfx as gfx
import zarr
from qtpy import QtWidgets
from wgpu.gui.qt import WgpuCanvas

from cellier.util.chunk import ChunkedArray3D, MultiScaleChunkedArray3D
from cellier.util.geometry import (
    frustum_planes_from_corners,
)

VOLUME_TEXTURE_SHAPE = (256, 256, 256)

root = zarr.open("multiscale_blobs.zarr")
multiscale_image = []
for level in range(5):
    multiscale_image.append(root[f"level_{level}"])

data_models = []
for level, image in enumerate(multiscale_image):
    scale = 2**level
    data_models.append(
        ChunkedArray3D(
            array_shape=image.shape,
            chunk_shape=image.chunks,
            scale=np.array((scale, scale, scale)),
            translation=np.array((0, 0, 0)),
        )
    )
multiscale_array_data = MultiScaleChunkedArray3D(scales=data_models)


class Main(QtWidgets.QWidget):
    """Main window."""

    def __init__(self):
        super().__init__(None)
        self.resize(640, 480)

        # Creat button and hook it up
        self._button = QtWidgets.QPushButton("Draw current view", self)
        self._button.clicked.connect(self._on_button_click)

        self._toggle_button = QtWidgets.QPushButton("Toggle main image", self)
        self._toggle_button.clicked.connect(self._on_toggle_image)

        # Create canvas, renderer and a scene object
        self._canvas = WgpuCanvas(parent=self)
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        self._camera = gfx.OrthographicCamera(110, 110)
        self._controller = gfx.OrbitController(
            self._camera, register_events=self._renderer
        )

        # make the data models
        self._multiscale_array = multiscale_array_data

        # volume_image = np.random.random((700, 700, 700)).astype(np.float32)
        self._setup_visuals()

        # Initialize the debugging visuals
        self.frustum_lines = None

        self._camera.show_object(self._points, view_dir=(0, 1, 0), up=(0, 0, 1))

        # Hook up the animate callback
        self._canvas.request_draw(self.animate)

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._button)
        layout.addWidget(self._toggle_button)
        layout.addWidget(self._canvas)

    def _setup_visuals(self):
        # for debugging put the full image in
        # tex = gfx.Texture(blob_image, dim=3)
        # self.full_img = gfx.Volume(
        #     gfx.Geometry(grid=tex),
        #     gfx.VolumeIsoMaterial(clim=(0, 1), map=gfx.cm.viridis, pick_write=False),
        # )
        # self._scene.add(self.full_img)
        self._points = gfx.Points(
            gfx.Geometry(
                positions=np.array([[0, 0, 0], [1024, 1024, 1024]], dtype=np.float32)
            ),
            gfx.PointsMaterial(size=10, color=(0, 1, 0.5, 1.0)),
        )
        self._image_box_world = gfx.BoxHelper(color="red")
        self._image_box_world.set_transform_by_object(self._points)
        self._scene.add(self._points)
        self._scene.add(self._image_box_world)

        self._multiscale_group = gfx.Group(name="multiscale")
        self.visuals_list = []
        self.bounding_boxes = []

        # add visual for each scale image
        for data_model in self._multiscale_array.scales:
            # make a visual for each scale
            tex = gfx.Texture(np.zeros(VOLUME_TEXTURE_SHAPE, dtype=np.float32), dim=3)
            vol = gfx.Volume(
                gfx.Geometry(grid=tex),
                gfx.VolumeIsoMaterial(clim=(0, 1), map=gfx.cm.plasma, pick_write=False),
            )
            vol.local.scale = data_model.scale
            vol.visible = False
            self.visuals_list.append(vol)
            self._multiscale_group.add(vol)

        self._scene.add(self._multiscale_group)

    def _on_button_click(self):
        """Callback to draw the chunks based on the current view.

        Eventually this would happen on the mouse move or camera update events.
        """
        # draw the camera frustum
        # frustum_edges = frustum_edges_from_corners(self._camera.frustum)
        # line_coordinates = frustum_edges.reshape(24, 3)

        # if self.frustum_lines is None:
        #     # if lines haven't been made, create them
        #     self.frustum_lines = gfx.Line(
        #         gfx.Geometry(positions=line_coordinates),
        #         gfx.LineSegmentMaterial(thickness=3),
        #     )
        #     self._scene.add(self.frustum_lines)
        # else:
        #     # otherwise, just update
        #     self.frustum_lines.geometry = gfx.Geometry(positions=line_coordinates)

        frustum_corners = self._camera.frustum

        scale = self._multiscale_array.scale_from_frustum(
            frustum_corners=frustum_corners,
            texture_shape=np.array(VOLUME_TEXTURE_SHAPE),
            width_factor=1.5,
        )
        print(f"scale: {scale.scale}")

        # clear current visuals
        self._clear_textures()

        for chunked_array, array_data, volume_visual in zip(
            self._multiscale_array.scales, multiscale_image, self.visuals_list
        ):
            if chunked_array is not scale:
                volume_visual.visible = False
            else:
                # transform the corners
                # todo do properly
                flat_corners = frustum_corners.reshape(8, 3)
                transformed_flat_corners = (
                    flat_corners / chunked_array.scale
                ) - chunked_array.translation
                transformed_corners = transformed_flat_corners.reshape(2, 4, 3)

                frustum_planes = frustum_planes_from_corners(transformed_corners)
                chunk_mask = chunked_array.chunks_in_frustum(
                    planes=frustum_planes, mode="any"
                )

                # update image texture
                texture = volume_visual.geometry.grid

                # get the chunks that need to be updated
                chunks_to_update = chunked_array.chunk_corners[chunk_mask]

                # get the lower-left bounds of the array
                n_chunks_to_update = chunks_to_update.shape[0]
                chunks_to_update_flat = chunks_to_update.reshape(
                    (n_chunks_to_update * 8, 3)
                )
                min_corner_all_local = np.min(chunks_to_update_flat, axis=0)
                min_corner_all_global = (
                    min_corner_all_local * chunked_array.scale
                ) + chunked_array.translation
                print(min_corner_all_local)
                print(min_corner_all_global)
                texture_shape = np.asarray(texture.data.shape)
                for chunk_corners in chunks_to_update:
                    # get the corners of the chunk in the array index coordinates
                    min_corner_array = chunk_corners[0]
                    max_corner_array = chunk_corners[7]

                    # get the corners of the chunk in the texture index coordinates
                    min_corner_texture = min_corner_array - min_corner_all_local
                    max_corner_texture = max_corner_array - min_corner_all_local

                    if np.any(max_corner_texture > texture_shape) or np.any(
                        min_corner_texture > texture_shape
                    ):
                        print(f"skipping: {min_corner_array}")
                        continue

                    texture.data[
                        min_corner_texture[2] : max_corner_texture[2],
                        min_corner_texture[1] : max_corner_texture[1],
                        min_corner_texture[0] : max_corner_texture[0],
                    ] = array_data[
                        min_corner_array[2] : max_corner_array[2],
                        min_corner_array[1] : max_corner_array[1],
                        min_corner_array[0] : max_corner_array[0],
                    ]
                    texture.update_range(
                        tuple(min_corner_array), tuple(chunked_array.chunk_shape)
                    )

                # update the visual transformation
                volume_visual.local.position = (
                    min_corner_all_global[0],
                    min_corner_all_global[1],
                    min_corner_all_global[2],
                )
                volume_visual.visible = True

        self._renderer.request_draw()

    def _clear_textures(self):
        for visual in self.visuals_list:
            texture = visual.geometry.grid
            texture.data[:, :, :] = np.zeros(VOLUME_TEXTURE_SHAPE, dtype=np.float32)
            texture.update_range((0, 0, 0), VOLUME_TEXTURE_SHAPE)

    def _on_toggle_image(self):
        if self.full_img.visible:
            self.full_img.visible = False
        else:
            self.full_img.visible = True

        self._renderer.request_draw()

    def animate(self):
        """Run the render loop."""
        self._renderer.render(self._scene, self._camera)


app = QtWidgets.QApplication([])
m = Main()
m.show()


if __name__ == "__main__":
    app.exec()
