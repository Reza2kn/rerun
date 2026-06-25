"""Log Genesis IMU data as scalar time series."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import rerun as rr

from genesis_rerun.sensors._arrays import select_env


def log(
    scene: Any,
    recording: Any,
    env_index: int,
    *,
    root_path: str = "world",
    sensor_handles: Sequence[Any] = (),
) -> None:
    """Log IMU-style namedtuple readings when present."""

    del scene
    for idx, sensor in enumerate(sensor_handles):
        if not hasattr(sensor, "read"):
            continue
        data = sensor.read()
        for field_name in ("lin_acc", "ang_vel", "mag"):
            if not hasattr(data, field_name):
                continue
            values = select_env(getattr(data, field_name), env_index)
            for axis, value in zip(("x", "y", "z"), values, strict=False):
                recording.log(
                    f"{root_path}/imu_{idx}/{field_name}/{axis}",
                    rr.Scalars(float(value)),
                )
