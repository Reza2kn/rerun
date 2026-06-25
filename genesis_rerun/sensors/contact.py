"""Log Genesis contact-force sensors as Rerun arrows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
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
    """Log sensors with `read()` values shaped like force vectors."""

    del scene
    for idx, sensor in enumerate(sensor_handles):
        if not hasattr(sensor, "read"):
            continue
        data = sensor.read()
        if hasattr(data, "_fields"):
            continue
        value = select_env(data, env_index)
        if value.shape[-1:] != (3,):
            continue
        origin = np.zeros(3, dtype=float)
        recording.log(
            f"{root_path}/contact_force_{idx}",
            rr.Arrows3D(origins=[origin], vectors=[value.astype(float)]),
        )
