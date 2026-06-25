"""The GenesisRerunLogger dispatcher ties the per-sensor adapters together."""

from __future__ import annotations

import pytest

from genesis_rerun import GenesisRerunLogger

from .conftest import FakeEntity, FakeLink, FakeRecording, FakeScene


def _scene_with_one_link() -> FakeScene:
    link = FakeLink("hand", pos=[0, 0, 1], quat_wxyz=[1, 0, 0, 0])
    return FakeScene(entities=[FakeEntity("panda", links=[link])])


def _logger(recording: FakeRecording, sensors: list[str] | None = None) -> GenesisRerunLogger:
    return GenesisRerunLogger(scene=_scene_with_one_link(), sensors=sensors, recording=recording)


def test_default_sensors_are_transforms_and_camera(recording) -> None:
    assert _logger(recording).sensors == ["transforms", "camera"]


def test_log_setup_declares_world_view_coordinates(recording) -> None:
    _logger(recording).log_setup()

    world_entries = [e for e in recording.entries if e.path == "world"]
    assert world_entries, "expected a static ViewCoordinates log on the world root"
    assert type(world_entries[0].archetype).__name__ == "ViewCoordinates"
    assert world_entries[0].kwargs.get("static") is True


def test_log_step_sets_both_timelines(recording) -> None:
    _logger(recording, sensors=["transforms"]).log_step(step=3, sim_time=0.05)

    timelines = {name for name, _ in recording.times}
    assert timelines == {"sim_step", "sim_time"}


def test_log_step_dispatches_to_enabled_sensor(recording) -> None:
    _logger(recording, sensors=["transforms"]).log_step(step=0, sim_time=0.0)

    assert recording.archetypes("Transform3D"), "transforms sensor should have logged a pose"


def test_unknown_sensor_name_raises(recording) -> None:
    logger = _logger(recording, sensors=["does_not_exist"])
    with pytest.raises(ValueError, match="Unknown Genesis/Rerun sensor logger"):
        logger.log_step(step=0, sim_time=0.0)
