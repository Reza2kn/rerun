"""Contact / IMU / tactile / lidar adapters turn Genesis sensor reads into Rerun."""

from __future__ import annotations

from genesis_rerun.sensors import contact, imu, lidar, tactile

from .conftest import FakeArraySensor, FakeFieldSensor


def test_contact_logs_force_vectors_as_arrows(recording) -> None:
    sensors = [FakeArraySensor([0.0, 0.0, -5.0])]
    contact.log(None, recording, env_index=0, sensor_handles=sensors)

    arrows = recording.archetypes("Arrows3D")
    assert len(arrows) == 1
    assert arrows[0].path == "world/contact_force_0"


def test_contact_ignores_non_vector_readings(recording) -> None:
    # A scalar (shape () not (3,)) reading must be skipped, not logged.
    sensors = [FakeArraySensor(1.0)]
    contact.log(None, recording, env_index=0, sensor_handles=sensors)
    assert recording.entries == []


def test_imu_logs_one_scalar_per_axis_per_channel(recording) -> None:
    sensors = [FakeFieldSensor(lin_acc=[0.1, 0.2, 9.8], ang_vel=[0.0, 0.0, 0.5])]
    imu.log(None, recording, env_index=0, sensor_handles=sensors)

    # lin_acc x/y/z + ang_vel x/y/z = 6 scalar channels.
    scalars = recording.archetypes("Scalars")
    assert len(scalars) == 6
    assert "world/imu_0/lin_acc/x" in recording.paths()
    assert "world/imu_0/ang_vel/z" in recording.paths()


def test_tactile_logs_min_mean_max_summaries(recording) -> None:
    sensors = [FakeFieldSensor(force=[1.0, 2.0, 3.0])]
    tactile.log(None, recording, env_index=0, sensor_handles=sensors)

    paths = recording.paths()
    assert "world/tactile_0/force/min" in paths
    assert "world/tactile_0/force/mean" in paths
    assert "world/tactile_0/force/max" in paths


def test_lidar_summarises_finite_distances(recording) -> None:
    sensors = [FakeFieldSensor(distances=[1.0, 2.0, float("inf"), 4.0])]
    lidar.log(None, recording, env_index=0, sensor_handles=sensors)

    paths = recording.paths()
    assert any(p.endswith("/distance_m/min") for p in paths)
    assert any(p.endswith("/distance_m/max") for p in paths)


def test_sensor_handles_without_read_are_skipped(recording) -> None:
    class NotASensor:
        pass

    contact.log(None, recording, env_index=0, sensor_handles=[NotASensor()])
    imu.log(None, recording, env_index=0, sensor_handles=[NotASensor()])
    assert recording.entries == []
