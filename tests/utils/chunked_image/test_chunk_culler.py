"""Tests for ChunkCuller (Phase B).

Uses shared fixtures from conftest.py and the make_box_frustum helper.
"""

import numpy as np

from cellier.utils.chunked_image._data_classes import ViewParameters

from .conftest import make_box_frustum


def test_all_chunks_visible(
    chunk_culler, scale_level_identity, identity_world_transform
):
    """T-B-01: Frustum completely encloses volume — all 64 chunks visible."""
    frustum = make_box_frustum(
        z_near=-10,
        z_far=150,
        y_min=-10,
        y_max=138,
        x_min=-10,
        x_max=138,
    )
    mask = chunk_culler.cull_chunks(
        scale_level_identity, frustum, identity_world_transform
    )
    assert mask.shape == (64,)
    assert mask.sum() == 64
    assert np.all(mask)


def test_no_chunks_visible(
    chunk_culler, scale_level_identity, identity_world_transform
):
    """T-B-02: Frustum entirely before the volume — 0 chunks visible."""
    frustum = make_box_frustum(
        z_near=-100,
        z_far=-5,
        y_min=-10,
        y_max=138,
        x_min=-10,
        x_max=138,
    )
    mask = chunk_culler.cull_chunks(
        scale_level_identity, frustum, identity_world_transform
    )
    assert mask.sum() == 0
    assert not np.any(mask)


def test_partial_visibility_upper_half(
    chunk_culler, scale_level_identity, identity_world_transform
):
    """T-B-03: Near plane at z=65 selects exactly the upper 32 chunks."""
    frustum = make_box_frustum(
        z_near=65,
        z_far=200,
        y_min=-10,
        y_max=138,
        x_min=-10,
        x_max=138,
    )
    mask = chunk_culler.cull_chunks(
        scale_level_identity, frustum, identity_world_transform
    )
    assert mask.sum() == 32
    assert not mask[:32].any()  # z_idx=0 and z_idx=1: all False
    assert mask[32:].all()  # z_idx=2 and z_idx=3: all True


def test_boundary_chunk_straddling(
    chunk_culler, scale_level_identity, identity_world_transform
):
    """T-B-04: Near plane at z=16 bisects first slab — included via 'any' semantics."""
    frustum = make_box_frustum(
        z_near=16,
        z_far=200,
        y_min=-10,
        y_max=138,
        x_min=-10,
        x_max=138,
    )
    mask = chunk_culler.cull_chunks(
        scale_level_identity, frustum, identity_world_transform
    )
    assert mask.sum() == 64
    assert mask[0]  # first chunk (z=[0,32]) straddles near plane but is included


def test_non_identity_world_transform(
    chunk_culler, scale_level_identity, scale2_world_transform
):
    """T-B-05: Scale-2 world transform — same physical region, all 64 chunks visible."""
    # After imap (÷2): z∈[-10,150], y∈[-10,138], x∈[-10,138] — same as T-B-01
    frustum_world = make_box_frustum(
        z_near=-20,
        z_far=300,
        y_min=-20,
        y_max=276,
        x_min=-20,
        x_max=276,
    )
    mask = chunk_culler.cull_chunks(
        scale_level_identity, frustum_world, scale2_world_transform
    )
    assert mask.sum() == 64
    assert np.all(mask)


def test_integration_with_chunk_selector(
    chunk_culler,
    chunk_selector,
    scale_level_identity,
    identity_world_transform,
    texture_config_64,
):
    """T-B-06: ChunkCuller output feeds into ChunkSelector without errors."""
    frustum = make_box_frustum(
        z_near=65,
        z_far=200,
        y_min=-10,
        y_max=138,
        x_min=-10,
        x_max=138,
    )
    mask = chunk_culler.cull_chunks(
        scale_level_identity, frustum, identity_world_transform
    )
    assert mask.sum() == 32  # sanity-check culler

    view_params = ViewParameters(
        frustum_corners=frustum,
        view_direction=np.array([1.0, 0.0, 0.0]),  # looking along +z
        near_plane_center=np.array([65.0, 64.0, 64.0]),
    )
    result = chunk_selector.select_chunks(
        scale_level=scale_level_identity,
        view_params=view_params,
        texture_config=texture_config_64,
        frustum_visible_chunks=mask,
    )

    assert result.n_selected_chunks > 0
    assert result.n_selected_chunks <= 32  # cannot exceed culled count
    assert result.selected_chunk_mask.shape == (64,)
