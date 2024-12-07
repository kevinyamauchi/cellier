"""Example showing progressive loading of volume data."""

import numpy as np
import pygfx as gfx
from qtpy import QtWidgets
from skimage.data import binary_blobs
from skimage.transform import resize
from wgpu.gui.qt import WgpuCanvas

from cellier.util.chunk import ChunkedArray3D, MultiScaleChunkedArray3D
from cellier.util.geometry import (
    frustum_planes_from_corners,
)

VOLUME_TEXTURE_SHAPE = (128, 128, 128)
blob_image = binary_blobs(
    length=128, n_dim=3, blob_size_fraction=0.2, volume_fraction=0.2
).astype(np.float32)

downscale_factor = 4
downscaled_transform_scale = (4, 4, 4)
downscaled_transform_translation = (1.5, 1.5, 1.5)
downscaled_blob = resize(blob_image, (32, 32, 32), order=0).astype(np.float32)

multiscale_image = [blob_image, downscaled_blob]


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

        # make the scene
        # volume_image = np.random.random((700, 700, 700)).astype(np.float32)
        self._setup_visuals()

        # make the grid of points
        self._chunk_data = ChunkedArray3D(
            array_shape=(128, 128, 128),
            chunk_shape=(4, 4, 4),
            scale=np.array((1, 1, 1)),
            translation=np.array((0, 0, 0)),
        )

        self._downscaled_chunk_data = ChunkedArray3D(
            array_shape=(32, 32, 32),
            chunk_shape=(4, 4, 4),
            scale=np.array(downscaled_transform_scale),
            translation=np.array(downscaled_transform_translation),
        )
        self._multiscale_array = MultiScaleChunkedArray3D(
            scales=[self._chunk_data, self._downscaled_chunk_data]
        )

        # Initialize the debugging visuals
        self.frustum_lines = None

        self._grid_points = self._chunk_data.chunk_centers.astype(np.float32)
        # self._points = gfx.Points(
        #     gfx.Geometry(positions=self._grid_points),
        #     gfx.PointsMaterial(size=10, color=(0, 1, 0.5, 1.0)),
        # )
        # self._scene.add(self._points)

        # axes = gfx.AxesHelper(5)
        # self._scene.add(axes)

        self._camera.show_object(self.full_img, view_dir=(0, 1, 0), up=(0, 0, 1))

        # Hook up the animate callback
        self._canvas.request_draw(self.animate)

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._button)
        layout.addWidget(self._toggle_button)
        layout.addWidget(self._canvas)

    def _setup_visuals(self):
        # for debugging put the full image in
        tex = gfx.Texture(blob_image, dim=3)
        self.full_img = gfx.Volume(
            gfx.Geometry(grid=tex),
            gfx.VolumeIsoMaterial(clim=(0, 1), map=gfx.cm.viridis, pick_write=False),
        )
        self._scene.add(self.full_img)

        self._multiscale_group = gfx.Group(name="multiscale")
        self.visuals_list = []
        self.bounding_boxes = []

        tex = gfx.Texture(np.zeros(VOLUME_TEXTURE_SHAPE, dtype=np.float32), dim=3)
        vol = gfx.Volume(
            gfx.Geometry(grid=tex),
            gfx.VolumeIsoMaterial(clim=(0, 1), map=gfx.cm.cividis, pick_write=False),
        )
        # vol.visible = False

        vol_bounding_box = gfx.BoxHelper(color="red")
        vol_bounding_box.set_transform_by_object(vol, space="local")
        # vol.add(vol_bounding_box)

        self.visuals_list.append(vol)
        self.bounding_boxes.append(vol_bounding_box)
        self._multiscale_group.add(vol)

        # add downscaled image
        tex = gfx.Texture(np.zeros(VOLUME_TEXTURE_SHAPE, dtype=np.float32), dim=3)
        downscaled_vol = gfx.Volume(
            gfx.Geometry(grid=tex),
            gfx.VolumeIsoMaterial(clim=(0, 1), map=gfx.cm.plasma, pick_write=False),
        )
        downscaled_vol.local.scale = downscaled_transform_scale
        downscaled_vol.local.position = downscaled_transform_translation
        downscaled_vol.visible = False

        downscaled_bounding_box = gfx.BoxHelper(color="blue")
        downscaled_bounding_box.set_transform_by_object(downscaled_vol, space="local")
        # downscaled_vol.add(downscaled_bounding_box)

        self.visuals_list.append(downscaled_vol)
        self.bounding_boxes.append(downscaled_bounding_box)
        self._multiscale_group.add(downscaled_vol)

        self._scene.add(self._multiscale_group)
        # self._scene.add(vol)
        # self._scene.add(downscaled_vol)

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

        for chunked_array, array_data, volume_visual, bounding_box_visual in zip(
            self._multiscale_array.scales,
            multiscale_image,
            self.visuals_list,
            self.bounding_boxes,
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

                for chunk_corners in chunks_to_update:
                    # get the corners of the chunk in the array index coordinates
                    min_corner_array = chunk_corners[0]
                    max_corner_array = chunk_corners[7]

                    # get the corners of the chunk in the texture index coordinates
                    min_corner_texture = min_corner_array - min_corner_all_local
                    max_corner_texture = max_corner_array - min_corner_all_local

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
                        tuple(min_corner_array), tuple(self._chunk_data.chunk_shape)
                    )

                # update the visual transformation
                volume_visual.local.position = (
                    min_corner_all_global[0],
                    min_corner_all_global[1],
                    min_corner_all_global[2],
                )
                volume_visual.visible = True
                bounding_box_visual.set_transform_by_object(
                    volume_visual, space="local"
                )

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
