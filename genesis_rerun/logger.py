"""High-level dispatcher for logging Genesis scenes into Rerun."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import rerun as rr

from genesis_rerun.assets.mjcf_loader import log_entity_meshes
from genesis_rerun.sensors import camera, contact, imu, lidar, tactile, transforms

SensorLogger = Callable[..., None]


@dataclass(slots=True)
class GenesisRerunLogger:
    """Log one Genesis environment into a Rerun recording stream."""

    scene: Any
    env_index: int = 0
    sensors: list[str] | None = None
    sensor_handles: list[Any] | None = None
    recording: Any = rr
    entity_path: str = "world"
    _dispatch: dict[str, SensorLogger] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dispatch = {
            "transforms": transforms.log,
            "camera": camera.log,
            "lidar": lidar.log,
            "contact": contact.log,
            "imu": imu.log,
            "tactile": tactile.log,
        }
        if self.sensors is None:
            self.sensors = ["transforms", "camera"]

    def log_setup(self) -> None:
        """Log static scene setup.

        Mesh logging is implemented separately because Genesis exposes visual geometry
        through entity/link-specific APIs that should be verified per installed version.
        """

        self.recording.log(self.entity_path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for entity in getattr(self.scene, "entities", []):
            log_entity_meshes(entity, self.recording, root_path=self.entity_path)

    def log_step(self, step: int, sim_time: float) -> None:
        """Set timelines and dispatch enabled sensor loggers."""

        self.recording.set_time("sim_step", sequence=step)
        self.recording.set_time("sim_time", duration=sim_time)
        for sensor_name in self.sensors or []:
            logger = self._dispatch.get(sensor_name)
            if logger is None:
                msg = f"Unknown Genesis/Rerun sensor logger: {sensor_name}"
                raise ValueError(msg)
            logger(
                self.scene,
                self.recording,
                self.env_index,
                root_path=self.entity_path,
                sensor_handles=self.sensor_handles or [],
            )
