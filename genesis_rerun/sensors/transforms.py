"""Log Genesis rigid-link poses as Rerun transforms."""

from __future__ import annotations

from typing import Any

import rerun as rr

from genesis_rerun.sensors._arrays import select_env


def genesis_wxyz_to_rerun_xyzw(quat: Any) -> Any:
    """Convert Genesis' wxyz quaternion convention to Rerun's xyzw convention."""

    return quat[[1, 2, 3, 0]]


def log(
    scene: Any,
    recording: Any,
    env_index: int,
    *,
    root_path: str = "world",
    sensor_handles: object = (),
) -> None:
    """Log every rigid link pose that exposes `get_pos` and `get_quat`."""

    del sensor_handles
    for entity in getattr(scene, "entities", []):
        entity_name = getattr(entity, "name", f"entity_{getattr(entity, 'idx', 0)}")
        for link in getattr(entity, "links", []):
            if not hasattr(link, "get_pos") or not hasattr(link, "get_quat"):
                continue
            raw_name = getattr(link, "name", f"link_{getattr(link, 'idx_local', 0)}")
            link_name = str(raw_name).strip("/") or "root"
            pos = select_env(link.get_pos(envs_idx=env_index), env_index)
            quat = select_env(link.get_quat(envs_idx=env_index), env_index)
            recording.log(
                f"{root_path}/{entity_name}/{link_name}",
                rr.Transform3D(
                    translation=pos,
                    rotation=rr.Quaternion(xyzw=genesis_wxyz_to_rerun_xyzw(quat)),
                    axis_length=0.0,
                ),
            )
