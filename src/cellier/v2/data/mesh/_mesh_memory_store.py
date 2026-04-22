# src/cellier/v2/data/mesh/_mesh_memory_store.py
from __future__ import annotations

import asyncio
from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator, model_validator

from cellier.v2.data._base_data_store import BaseDataStore
from cellier.v2.data.mesh._mesh_requests import MeshData, MeshSliceRequest

# Placeholder returned when the slab contains no surviving faces.
# A single degenerate triangle avoids the "empty geometry is illegal"
# restriction in pygfx.
_PLACEHOLDER_POSITIONS = np.zeros((3, 3), dtype=np.float32)
_PLACEHOLDER_INDICES = np.array([[0, 1, 2]], dtype=np.int32)
_PLACEHOLDER_NORMALS = np.tile([0.0, 0.0, 1.0], (3, 1)).astype(np.float32)


def _compute_vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute area-weighted per-vertex normals from triangle soup.

    Parameters
    ----------
    positions : np.ndarray
        (n_vertices, 3) float32 vertex positions.
    indices : np.ndarray
        (n_faces, 3) int32 triangle face indices.

    Returns
    -------
    np.ndarray
        (n_vertices, 3) float32 unit normals.  Degenerate vertices
        (zero accumulated normal) get [0, 0, 1].
    """
    e1 = positions[indices[:, 1]] - positions[indices[:, 0]]
    e2 = positions[indices[:, 2]] - positions[indices[:, 0]]
    face_normals = np.cross(e1, e2)  # area-weighted, (n_faces, 3)

    vertex_normals = np.zeros_like(positions)
    np.add.at(vertex_normals, indices[:, 0], face_normals)
    np.add.at(vertex_normals, indices[:, 1], face_normals)
    np.add.at(vertex_normals, indices[:, 2], face_normals)

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    safe = norms > 0
    vertex_normals = np.where(
        safe, vertex_normals / np.where(safe, norms, 1.0), [0.0, 0.0, 1.0]
    )
    return vertex_normals.astype(np.float32)


class MeshMemoryStore(BaseDataStore):
    """In-memory triangle mesh data store.

    Parameters
    ----------
    positions : np.ndarray
        (n_vertices, 3) float32 vertex positions.
    indices : np.ndarray
        (n_faces, 3) int32 triangle face indices.
        **Must be int32** — pygfx rejects int64 at upload time.
        int64 input is coerced silently by the validator.
    normals : np.ndarray | None
        (n_vertices, 3) float32 per-vertex normals.  Auto-computed
        from positions and indices if None.
    colors : np.ndarray | None
        Per-vertex (n_vertices, 4) or per-face (n_faces, 4) float32
        RGBA.  Layout inferred from shape.
    name : str
        Human-readable label.
    """

    store_type: Literal["mesh_memory"] = "mesh_memory"
    name: str = "mesh_memory_store"
    positions: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    colors: np.ndarray | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Validators — run in declaration order (mode="before" first)
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _compute_normals_if_missing(cls, data: Any) -> Any:
        """Auto-compute normals before pydantic processes any fields.

        Uses mode="before" so the result is injected into the raw input
        dict, avoiding object.__setattr__ on a psygnal EventedModel.
        """
        if not isinstance(data, dict):
            return data
        if data.get("normals") is not None:
            return data

        raw_pos = data.get("positions")
        raw_idx = data.get("indices")
        if raw_pos is None or raw_idx is None:
            return data  # missing fields — let field validators error

        pos = np.asarray(raw_pos, dtype=np.float32)
        idx = np.asarray(raw_idx, dtype=np.int32)
        data["normals"] = _compute_vertex_normals(pos, idx)
        return data

    @field_validator("positions", mode="before")
    @classmethod
    def _coerce_positions(cls, v: Any) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32))

    @field_validator("indices", mode="before")
    @classmethod
    def _coerce_indices(cls, v: Any) -> np.ndarray:
        """Coerce to int32 — pygfx rejects int64 index buffers."""
        return np.ascontiguousarray(np.asarray(v, dtype=np.int32))

    @field_validator("normals", mode="before")
    @classmethod
    def _coerce_normals(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32))

    @field_validator("colors", mode="before")
    @classmethod
    def _coerce_colors(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        return np.ascontiguousarray(np.asarray(v, dtype=np.float32))

    # ------------------------------------------------------------------
    # Serializers
    # ------------------------------------------------------------------

    @field_serializer("positions")
    def _ser_positions(self, v: np.ndarray, _info: Any) -> list:
        return v.tolist()

    @field_serializer("indices")
    def _ser_indices(self, v: np.ndarray, _info: Any) -> list:
        return v.tolist()

    @field_serializer("normals")
    def _ser_normals(self, v: np.ndarray | None, _info: Any) -> list | None:
        return v.tolist() if v is not None else None

    @field_serializer("colors")
    def _ser_colors(self, v: np.ndarray | None, _info: Any) -> list | None:
        return v.tolist() if v is not None else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return self.positions.shape[0]

    @property
    def n_faces(self) -> int:
        return self.indices.shape[0]

    @property
    def colors_mode(self) -> str:
        """``'vertex'``, ``'face'``, or ``'none'``."""
        if self.colors is None:
            return "none"
        return "face" if self.colors.shape[0] == self.n_faces else "vertex"

    # ------------------------------------------------------------------
    # Async data access — three checkpoints for cancellability
    # ------------------------------------------------------------------

    async def get_data(self, request: MeshSliceRequest) -> MeshData:
        """Return slab-filtered, reindexed mesh data for *request*.

        Checkpoints
        -----------
        A  After Phase 1 (slab mask built, no reindexing yet).
           Fires if slider moved before reindexing begins.
        B  After Phase 2 (reindex complete, before projection).
           Fires if slider moved before projection.

        If CancelledError fires at either checkpoint the callback is
        never called, preventing stale geometry from reaching the GPU.

        Inclusion rule
        -------------
        A face survives only when **all** of its vertices satisfy
        ``slice_index - thickness <= coord <= slice_index + thickness``
        on every sliced axis.  This is the mesh analogue of the lines
        store's "both endpoints must pass" rule and avoids projecting
        off-slice vertices onto the slice plane with the wrong colors.

        Parameters
        ----------
        request : MeshSliceRequest
            Built by GFXMeshMemoryVisual.build_slice_request[_2d].

        Returns
        -------
        MeshData
            Filtered, reindexed, projected mesh ready for GPU upload.
            ``is_empty=True`` when the slab contained no faces.
        """
        positions = self.positions  # (n_vertices, 3)
        indices = self.indices  # (n_faces, 3)
        normals = self.normals  # (n_vertices, 3)
        colors = self.colors
        n_vertices = positions.shape[0]
        displayed = list(request.displayed_axes)

        # ── Phase 1: build slab mask ─────────────────────────────────
        face_mask = np.ones(self.n_faces, dtype=bool)

        for axis, idx in request.slice_indices.items():
            lo = float(idx) - request.thickness
            hi = float(idx) + request.thickness
            # Include face only if ALL vertices are in the slab on this axis.
            vertex_in = (positions[:, axis] >= lo) & (positions[:, axis] <= hi)
            face_mask &= vertex_in[indices].all(axis=1)

        # ── Checkpoint A ─────────────────────────────────────────────
        await asyncio.sleep(0)

        # ── Phase 2: reindex surviving faces ─────────────────────────
        surviving = indices[face_mask]  # (n_surv, 3)

        if surviving.shape[0] == 0:
            # Empty slab — return placeholder so the node stays valid.
            return MeshData(
                request_id=request.slice_request_id,
                positions=_PLACEHOLDER_POSITIONS[:, displayed],
                indices=_PLACEHOLDER_INDICES,
                normals=_PLACEHOLDER_NORMALS[:, displayed],
                colors=None,
                color_mode="vertex",
                is_empty=True,
            )

        unique_old = np.unique(surviving.ravel())
        remap = np.full(n_vertices, -1, dtype=np.int32)
        remap[unique_old] = np.arange(len(unique_old), dtype=np.int32)

        new_positions = positions[unique_old]  # (n_surv_v, 3)
        new_normals = normals[unique_old]  # (n_surv_v, 3)
        new_indices = remap[surviving]  # (n_surv_f, 3)

        if colors is not None:
            if self.colors_mode == "face":
                new_colors = colors[face_mask]  # (n_surv_f, 4)
                color_mode = "face"
            else:
                new_colors = colors[unique_old]  # (n_surv_v, 4)
                color_mode = "vertex"
        else:
            new_colors = None
            color_mode = "vertex"

        # ── Checkpoint B ─────────────────────────────────────────────
        await asyncio.sleep(0)

        # ── Phase 3: project onto displayed axes ─────────────────────
        proj_positions = new_positions[:, displayed]
        proj_normals = new_normals[:, displayed]

        return MeshData(
            request_id=request.slice_request_id,
            positions=proj_positions,
            indices=new_indices,
            normals=proj_normals,
            colors=new_colors,
            color_mode=color_mode,
            is_empty=False,
        )
