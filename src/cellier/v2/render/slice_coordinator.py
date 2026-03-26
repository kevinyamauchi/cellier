"""SliceCoordinator — thin orchestrator that drives the async reslicing cycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.v2.data.image import MultiscaleZarrDataStore
    from cellier.v2.render._requests import ReslicingRequest
    from cellier.v2.render._scene_config import VisualRenderConfig
    from cellier.v2.render.scene_manager import SceneManager
    from cellier.v2.slicer import AsyncSlicer


class SliceCoordinator:
    """Thin orchestrator owned by ``RenderManager``.

    Given a ``ReslicingRequest``, it looks up the target ``SceneManager``,
    runs the synchronous planning phase, cancels in-flight tasks for the
    affected visuals, and submits new async load tasks.

    One ``AsyncSlicer`` task maps to one ``(scene_id, visual_id)`` pair.
    A ``dict`` keyed by ``(scene_id, visual_id)`` tracks active slice IDs
    so that per-visual cancellation can cancel only the affected task while
    leaving other visuals in the same scene running.

    Parameters
    ----------
    scenes : dict[UUID, SceneManager]
        Shared scene registry from ``RenderManager``.
    slicer : AsyncSlicer
        Shared async slicer instance.
    data_stores : dict[UUID, MultiscaleZarrDataStore]
        Mapping of ``visual_model_id`` to the data store for that visual.
    """

    def __init__(
        self,
        scenes: dict[UUID, SceneManager],
        slicer: AsyncSlicer,
        data_stores: dict[UUID, MultiscaleZarrDataStore],
    ) -> None:
        self._scenes = scenes
        self._slicer = slicer
        self._data_stores = data_stores
        self._active_slice_ids: dict[tuple[UUID, UUID], UUID] = {}

    def submit(
        self,
        request: ReslicingRequest,
        visual_configs: dict[UUID, VisualRenderConfig],
    ) -> None:
        """Execute the full reslicing cycle for the scene in ``request.scene_id``.

        Cancels in-flight tasks for visuals that will be re-submitted before
        running the synchronous planning phase.

        Parameters
        ----------
        request : ReslicingRequest
            The reslicing request to process.
        visual_configs : dict[UUID, VisualRenderConfig]
            Per-visual render configuration.
        """
        scene_manager = self._scenes[request.scene_id]

        if request.target_visual_ids is not None:
            to_cancel: frozenset[UUID] = request.target_visual_ids
        else:
            to_cancel = frozenset(scene_manager.visual_ids)

        for visual_id in to_cancel:
            self.cancel_visual(request.scene_id, visual_id)

        requests_by_visual = scene_manager.build_slice_requests(request, visual_configs)

        is_2d = scene_manager.dim == "2d"

        for visual_id, chunk_requests in requests_by_visual.items():
            visual = scene_manager.get_visual(visual_id)
            data_store = self._data_stores[visual_id]

            # Use the appropriate callback for the scene dimensionality.
            callback = visual.on_data_ready_2d if is_2d else visual.on_data_ready

            slice_id = self._slicer.submit(
                chunk_requests,
                fetch_fn=data_store.get_data,
                callback=callback,
                consumer_id=str(visual_id),
            )
            if slice_id is not None:
                self._active_slice_ids[(request.scene_id, visual_id)] = slice_id

    def cancel_scene(self, scene_id: UUID) -> None:
        """Cancel all in-flight tasks for a scene.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene whose tasks should be cancelled.
        """
        visual_ids = [
            v_id
            for (s_id, v_id) in list(self._active_slice_ids.keys())
            if s_id == scene_id
        ]
        for visual_id in visual_ids:
            self.cancel_visual(scene_id, visual_id)

    def cancel_visual(self, scene_id: UUID, visual_id: UUID) -> None:
        """Cancel the in-flight task for one visual.

        Also calls ``visual.cancel_pending()`` or ``cancel_pending_2d()``
        to release any GPU slots reserved during the last planning phase
        that were never committed.

        Parameters
        ----------
        scene_id : UUID
            ID of the scene containing the visual.
        visual_id : UUID
            ID of the visual to cancel.
        """
        key = (scene_id, visual_id)
        slice_id = self._active_slice_ids.pop(key, None)
        if slice_id is not None:
            self._slicer.cancel(slice_id)

        scene = self._scenes.get(scene_id)
        if scene is not None:
            try:
                visual = scene.get_visual(visual_id)
                if scene.dim == "2d":
                    visual.cancel_pending_2d()
                else:
                    visual.cancel_pending()
            except KeyError:
                pass

    # ── Dormant EventBus handler stubs ───────────────────────────────────

    def _on_dims_changed(self, event) -> None:
        """Dormant stub — not yet subscribed to the bus."""
        pass

    def _on_appearance_changed(self, event) -> None:
        """Dormant stub — not yet subscribed to the bus."""
        pass
