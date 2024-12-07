"""Example showing progressive loading of volume data."""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import current_thread

import numpy as np
import pygfx as gfx
import zarr
from qtpy import QtWidgets
from superqt import ensure_main_thread
from wgpu.gui.qt import WgpuCanvas

from cellier.models.data_stores.image import MultiScaleImageZarrStore
from cellier.util.chunk import (
    ChunkedArray3D,
    ImageDataStoreChunk,
    MultiScaleChunkedArray3D,
)
from cellier.util.geometry import (
    frustum_planes_from_corners,
)

VOLUME_TEXTURE_SHAPE = (256, 256, 256)

root = zarr.open("multiscale_blobs.zarr")
multiscale_image_arrays = []
for level in range(5):
    multiscale_image_arrays.append(root[f"level_{level}"])

data_models = []
for level, image in enumerate(multiscale_image_arrays):
    scale = 2**level
    data_models.append(
        ChunkedArray3D(
            array_shape=image.shape,
            chunk_shape=image.chunks,
            scale=np.array((scale, scale, scale)),
            translation=np.array((0, 0, 0)),
        )
    )
multiscale_array_data_model = MultiScaleChunkedArray3D(scales=data_models)

multiscale_store = MultiScaleImageZarrStore(
    root_path="multiscale_blobs.zarr",
    scale_paths=[f"level_{scale_index}" for scale_index in range(5)],
    scales=[
        (2**scale_index, 2**scale_index, 2**scale_index) for scale_index in range(5)
    ],
    translations=[(0, 0, 0) for scale_index in range(5)],
)


@dataclass
class ChunkRequest:
    """Data to get a chunk of data."""

    scale_index: int
    chunk_start: np.ndarray
    chunk_end: np.ndarray
    texture_start: np.ndarray
    texture_end: np.ndarray


@dataclass
class ChunkResponse:
    """Data to get a chunk of data."""

    scale_index: int
    chunk_start: np.ndarray
    chunk_end: np.ndarray
    texture_start: np.ndarray
    texture_end: np.ndarray
    array: np.ndarray


@dataclass
class MultiScaleImage:
    """Store a multiscale image."""

    arrays: list[np.ndarray]

    def get_chunk(self, chunk_request: ChunkRequest) -> ChunkResponse:
        """Get a chunk."""
        array = self.arrays[chunk_request.scale_index]

        start_indices = chunk_request.chunk_start
        end_indices = chunk_request.chunk_end

        return ChunkResponse(
            scale_index=chunk_request.scale_index,
            chunk_start=chunk_request.chunk_start,
            chunk_end=chunk_request.chunk_end,
            texture_start=chunk_request.texture_start,
            texture_end=chunk_request.texture_end,
            array=array[
                start_indices[0] : end_indices[0],
                start_indices[1] : end_indices[1],
                start_indices[2] : end_indices[2],
            ],
        )


multiscale_image = MultiScaleImage(arrays=multiscale_image_arrays)


class Main(QtWidgets.QWidget):
    """Main window."""

    def __init__(self, synchronous_slicing: bool = False):
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

        # remap panning to shift + lmb
        pan_value = self._controller.controls.pop("mouse2")
        self._controller.controls["shift+mouse1"] = pan_value

        # make the data models
        self._multiscale_array = multiscale_array_data_model

        # volume_image = np.random.random((700, 700, 700)).astype(np.float32)
        self._setup_visuals()

        # flag for synchronous slicing
        self.synchronous_slicing = synchronous_slicing

        # make the threadpool
        max_workers = 1
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_futures = []
        self._futures_to_ignore = []

        # Initialize the debugging visuals
        self.frustum_lines = None

        self._camera.show_object(self._points, view_dir=(0, 1, 0), up=(0, 0, 1))

        print(self._camera.get_state())

        # Hook up the animate callback
        self._canvas.request_draw(self.animate)

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._button)
        layout.addWidget(self._toggle_button)
        layout.addWidget(self._canvas)

    @ensure_main_thread()
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

    @ensure_main_thread()
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

        width = self._camera.width
        height = self._camera.height
        print(f"camera width: {width} camera height: {height}")

        frustum_corners = self._camera.frustum

        near_plane = frustum_corners[0]
        width_logical, height_logical = self._renderer.logical_size
        width_world = np.linalg.norm(near_plane[1, :] - near_plane[0, :])
        height_world = np.linalg.norm(near_plane[3, :] - near_plane[0, :])
        width_size_world = width_world / width_logical
        height_size_world = height_world / height_logical
        print(f"width world: {width_world}, height world: {height_world}")
        print(f"width logical: {width_logical}, height logical: {height_logical}")
        print(
            f"width size world: {width_size_world},"
            f"height size world: {height_size_world}"
        )

        scale = self._multiscale_array.scale_from_frustum(
            frustum_corners=frustum_corners,
            width_logical=width_logical,
            height_logical=height_logical,
            texture_shape=np.array(VOLUME_TEXTURE_SHAPE),
            width_factor=1.5,
            method="logical_pixel_size",
        )
        print(f"scale: {scale.scale}")

        # clear current visuals
        # self._clear_textures()

        for scale_index, chunked_array, array_data, volume_visual in zip(
            range(self._multiscale_array.n_scales),
            self._multiscale_array.scales,
            multiscale_image.arrays,
            self.visuals_list,
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

                start_time = time.time()
                # get the chunks that need to be updated
                chunks_to_update = chunked_array.chunk_corners[chunk_mask]

                # get the lower-left bounds of the array
                n_chunks_to_update = chunks_to_update.shape[0]
                chunks_to_update_flat = chunks_to_update.reshape(
                    (n_chunks_to_update * 8, 3)
                )
                min_corner_all_local = np.min(chunks_to_update_flat, axis=0)
                max_corner_all_local = np.max(chunks_to_update_flat, axis=0)
                min_corner_all_global = (
                    min_corner_all_local * chunked_array.scale
                ) + chunked_array.translation
                new_texture_shape = max_corner_all_local - min_corner_all_local
                print(f"min corner local: {min_corner_all_local}")
                print(f"texture shape: {new_texture_shape}")

                print(f"time to find chunks: {time.time() - start_time}")

                old_texture = volume_visual.geometry.grid

                texture = gfx.Texture(
                    np.zeros(
                        (
                            new_texture_shape[2],
                            new_texture_shape[1],
                            new_texture_shape[0],
                        ),
                        dtype=np.float32,
                    ),
                    dim=3,
                    chunk_size=chunked_array.chunk_shape,
                )
                volume_visual.geometry = gfx.Geometry(grid=texture)
                texture_shape = np.asarray(texture.data.shape)[[2, 1, 0]]
                print(f"made new texture: {texture_shape}")
                del old_texture

                if self.synchronous_slicing:
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
                            tuple(min_corner_texture), tuple(chunked_array.chunk_shape)
                        )

                        # update the visual transformation
                        volume_visual.local.position = (
                            min_corner_all_global[0],
                            min_corner_all_global[1],
                            min_corner_all_global[2],
                        )
                        volume_visual.visible = True

                        self._renderer.request_draw()
                else:
                    # update the visual transformation
                    volume_visual.local.position = (
                        min_corner_all_global[0],
                        min_corner_all_global[1],
                        min_corner_all_global[2],
                    )
                    volume_visual.visible = True

                    print(f"{len(chunks_to_update)} to update")
                    self._futures_received = 0
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

                        chunk_request = ImageDataStoreChunk(
                            resolution_level=scale_index,
                            array_coordinate_start=min_corner_array[[2, 1, 0]],
                            array_coordinate_end=max_corner_array[[2, 1, 0]],
                            texture_coordinate_start=min_corner_texture,
                            scene_id="test",
                            visual_id="test",
                        )
                        chunk_future = self._thread_pool.submit(
                            multiscale_store.get_slice, chunk_request
                        )
                        chunk_future.add_done_callback(self._on_slice_response)

                        self._pending_futures.append(chunk_future)

                    print(f"{len(self._pending_futures)} submitted")

    @ensure_main_thread
    def _on_slice_response(self, future: Future[ChunkResponse]):
        if future.cancelled():
            # if the future was cancelled, return early
            return
        if future in self._futures_to_ignore:
            print("ignoring")
            self._futures_to_ignore.remove(future)
            return

        response = future.result()

        min_corner_texture = np.asarray(response.texture_start_index)
        max_corner_texture = min_corner_texture + np.asarray(response.data.shape)

        volume_visual = self.visuals_list[response.resolution_level]
        texture = volume_visual.geometry.grid

        texture.data[
            min_corner_texture[2] : max_corner_texture[2],
            min_corner_texture[1] : max_corner_texture[1],
            min_corner_texture[0] : max_corner_texture[0],
        ] = response.data
        texture.update_range(
            tuple(min_corner_texture), tuple(max_corner_texture - min_corner_texture)
        )

        self._pending_futures.remove(future)
        del future

        self._futures_received += 1
        current_thread_string = str(current_thread())
        if "MainThread" not in current_thread_string:
            print(f"{self._futures_received}: {current_thread()}")

        self._renderer.request_draw()

    @ensure_main_thread()
    def _clear_textures(self):
        for visual in self.visuals_list:
            texture = visual.geometry.grid
            texture.data[:, :, :] = np.zeros(VOLUME_TEXTURE_SHAPE, dtype=np.float32)
            texture.update_range((0, 0, 0), VOLUME_TEXTURE_SHAPE)

    def _on_toggle_image(self):
        if self.synchronous_slicing:
            self.synchronous_slicing = False
        else:
            self.synchronous_slicing = True

        self._renderer.request_draw()

    def animate(self):
        """Run the render loop."""
        self._renderer.render(self._scene, self._camera)


app = QtWidgets.QApplication([])
m = Main()
m.show()


if __name__ == "__main__":
    app.exec()
