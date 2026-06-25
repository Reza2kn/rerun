"""Log Genesis tactile and field sensors as tensors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import rerun as rr

from genesis_rerun.sensors._arrays import select_env


def _log_numeric_summary(recording: Any, path: str, value: object, env_index: int) -> None:
    values = select_env(value, env_index).reshape(-1)
    if values.size == 0:
        return
    recording.log(f"{path}/min", rr.Scalars(float(values.min())))
    recording.log(f"{path}/mean", rr.Scalars(float(values.mean())))
    recording.log(f"{path}/max", rr.Scalars(float(values.max())))


def log(
    scene: Any,
    recording: Any,
    env_index: int,
    *,
    root_path: str = "world",
    sensor_handles: Sequence[Any] = (),
) -> None:
    """Log richer sensor readings as Rerun tensors."""

    del scene
    for idx, sensor in enumerate(sensor_handles):
        if not hasattr(sensor, "read"):
            continue
        data = sensor.read()
        for field_name in ("penetration", "force", "distances", "temperature"):
            if not hasattr(data, field_name):
                continue
            _log_numeric_summary(
                recording,
                f"{root_path}/tactile_{idx}/{field_name}",
                getattr(data, field_name),
                env_index,
            )
