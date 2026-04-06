"""Tests for RenderManagerConfig and related config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellier.v2.render._config import (
    CameraConfig,
    RenderManagerConfig,
    SlicingConfig,
    TemporalAccumulationConfig,
)
from cellier.v2.render.render_manager import RenderManager


def test_default_config_values():
    config = RenderManagerConfig()
    assert config.slicing.batch_size == 8
    assert config.slicing.render_every == 1
    assert config.temporal.enabled is True
    assert config.temporal.alpha == 0.1
    assert config.camera.reslice_enabled is True
    assert config.camera.settle_threshold_s == 0.3


def test_json_roundtrip():
    config = RenderManagerConfig(
        slicing=SlicingConfig(batch_size=32, render_every=4),
        temporal=TemporalAccumulationConfig(enabled=False, alpha=0.05),
        camera=CameraConfig(reslice_enabled=False, settle_threshold_s=0.5),
    )
    json_str = config.model_dump_json()
    config2 = RenderManagerConfig.model_validate_json(json_str)
    assert config2.model_dump_json() == json_str


def test_temporal_alpha_setter_updates_config():
    manager = RenderManager(config=RenderManagerConfig())
    manager.temporal_alpha = 0.02
    assert manager.config.temporal.alpha == 0.02


def test_temporal_enabled_setter_updates_config():
    manager = RenderManager(config=RenderManagerConfig())
    manager.temporal_enabled = False
    assert manager.config.temporal.enabled is False


def test_batch_size_must_be_positive():
    with pytest.raises(ValidationError):
        SlicingConfig(batch_size=0)


def test_render_every_must_be_positive():
    with pytest.raises(ValidationError):
        SlicingConfig(render_every=0)


def test_alpha_must_be_in_range():
    with pytest.raises(ValidationError):
        TemporalAccumulationConfig(alpha=0.0)
    with pytest.raises(ValidationError):
        TemporalAccumulationConfig(alpha=1.1)


def test_settle_threshold_must_be_positive():
    with pytest.raises(ValidationError):
        CameraConfig(settle_threshold_s=0.0)
