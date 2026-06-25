"""Log Genesis raycaster-style point clouds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import rerun as rr

from genesis_rerun.sensors._arrays import select_env


def _log_distance_summary(recording: Any, path: str, distances: object, env_index: int) -> None:
    values = select_env(distances, env_index).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    recording.log(f"{path}/distance_m/min", rr.Scalars(float(values.min())))
    recording.log(f"{path}/distance_m/mean", rr.Scalars(float(values.mean())))
    recording.log(f"{path}/distance_m/max", rr.Scalars(float(values.max())))


def log(
    scene: Any,
    recording: Any,
    env_index: int,
    *,
    root_path: str = "world",
    sensor_handles: Sequence[Any] = (),
) -> None:
    """Log sensor readings with a `points` field as Rerun point clouds."""

    del scene
    for idx, sensor in enumerate(sensor_handles):
        if not hasattr(sensor, "read"):
            continue
        data = sensor.read()
        sensor_name = type(sensor).__name__.replace("Sensor", "").lower()
        sensor_path = f"{root_path}/{sensor_name}_{idx}"
        if hasattr(data, "distances"):
            _log_distance_summary(recording, sensor_path, data.distances, env_index)
