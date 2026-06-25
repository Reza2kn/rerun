"""Log Genesis visualization cameras as Rerun images and derived point clouds."""

from __future__ import annotations

from typing import Any

import numpy as np
import rerun as rr

from genesis_rerun.sensors._arrays import as_numpy


def select_camera_env(value: object, env_index: int, *, unbatched_ndim: int) -> object:
    """Select env only when Genesis returns a leading environment dimension."""

    arr = as_numpy(value)
    if arr.ndim == unbatched_ndim + 1 and arr.shape[0] > env_index:
        return arr[env_index]
    return arr


def _camera_value(cam: Any, names: tuple[str, ...], fallback: Any) -> Any:
    for name in names:
        if hasattr(cam, name):
            value = getattr(cam, name)
            if callable(value):
                try:
                    return value()
                except TypeError:
                    continue
            return value
    return fallback


def _camera_pose(cam: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(
        as_numpy(_camera_value(cam, ("get_pos", "pos", "_pos"), (0.0, 0.0, 1.0))),
        dtype=float,
    )
    lookat = np.asarray(
        as_numpy(_camera_value(cam, ("get_lookat", "lookat", "_lookat"), (0.0, 0.0, 0.0))),
        dtype=float,
    )
    up = np.asarray(
        as_numpy(_camera_value(cam, ("get_up", "up", "_up"), (0.0, 0.0, 1.0))),
        dtype=float,
    )
    return pos.reshape(-1)[:3], lookat.reshape(-1)[:3], up.reshape(-1)[:3]


def _depth_to_world_points(
    depth: np.ndarray, cam: Any, *, stride: int, fov_degrees: float
) -> np.ndarray:
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2:
        return np.empty((0, 3), dtype=float)

    height, width = depth.shape
    if height == 0 or width == 0:
        return np.empty((0, 3), dtype=float)

    ys = np.arange(0, height, stride)
    xs = np.arange(0, width, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth[grid_y, grid_x].reshape(-1)
    valid = np.isfinite(z) & (z > 0.05) & (z < 6.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=float)

    pos, lookat, up = _camera_pose(cam)
    forward = lookat - pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-9:
        return np.empty((0, 3), dtype=float)
    forward = forward / forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0], dtype=float) if right_norm < 1e-9 else right / right_norm
    true_up = np.cross(right, forward)
    true_up = true_up / max(np.linalg.norm(true_up), 1e-9)

    aspect = width / max(float(height), 1.0)
    tan_y = np.tan(np.deg2rad(float(fov_degrees)) * 0.5)
    tan_x = tan_y * aspect
    x = ((grid_x.reshape(-1) + 0.5) / width - 0.5) * 2.0 * tan_x
    y = (0.5 - (grid_y.reshape(-1) + 0.5) / height) * 2.0 * tan_y
    dirs = forward[None, :] + x[:, None] * right[None, :] + y[:, None] * true_up[None, :]
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    return pos[None, :] + dirs[valid] * z[valid, None]


def _log_point_clouds_from_depth(
    recording: Any,
    root_path: str,
    cam: Any,
    depth: np.ndarray,
    *,
    camera_index: int,
) -> None:
    if camera_index != 0:
        return

    if hasattr(cam, "render_pointcloud"):
        try:
            point_grid, mask = cam.render_pointcloud(world_frame=True)
            point_grid = np.asarray(point_grid, dtype=float)
            mask = np.asarray(mask, dtype=bool)
        except Exception:
            point_grid = np.empty((0, 0, 3), dtype=float)
            mask = np.empty((0, 0), dtype=bool)

        grid_is_valid = (
            point_grid.ndim == 3
            and point_grid.shape[-1] == 3
            and mask.shape == point_grid.shape[:2]
        )
        if grid_is_valid:
            depth_stride = 8
            lidar_stride = 24
            depth_points = point_grid[::depth_stride, ::depth_stride, :].reshape(-1, 3)
            depth_mask = mask[::depth_stride, ::depth_stride].reshape(-1)
            lidar_points = point_grid[::lidar_stride, ::lidar_stride, :].reshape(-1, 3)
            lidar_mask = mask[::lidar_stride, ::lidar_stride].reshape(-1)

            depth_points = depth_points[depth_mask & np.all(np.isfinite(depth_points), axis=1)]
            lidar_points = lidar_points[lidar_mask & np.all(np.isfinite(lidar_points), axis=1)]
            if depth_points.size:
                recording.log(
                    f"{root_path}/depth_point_cloud/points",
                    rr.Points3D(depth_points, radii=0.004, colors=[82, 176, 255, 190]),
                )
            if lidar_points.size:
                recording.log(
                    f"{root_path}/lidar_point_cloud/points",
                    rr.Points3D(lidar_points, radii=0.012, colors=[255, 210, 72, 255]),
                )
            return

    fov = float(_camera_value(cam, ("fov", "_fov"), 55.0))
    depth_points = _depth_to_world_points(depth, cam, stride=10, fov_degrees=fov)
    lidar_points = _depth_to_world_points(depth, cam, stride=28, fov_degrees=fov)
    if depth_points.size:
        recording.log(
            f"{root_path}/depth_point_cloud/points",
            rr.Points3D(depth_points, radii=0.004, colors=[82, 176, 255, 190]),
        )
    if lidar_points.size:
        recording.log(
            f"{root_path}/lidar_point_cloud/points",
            rr.Points3D(lidar_points, radii=0.011, colors=[255, 210, 72, 255]),
        )


def log(
    scene: Any,
    recording: Any,
    env_index: int,
    *,
    root_path: str = "world",
    sensor_handles: object = (),
) -> None:
    """Render and log cameras attached to the Genesis visualizer."""

    del sensor_handles
    cameras = getattr(getattr(scene, "visualizer", None), "cameras", None)
    if cameras is None:
        cameras = getattr(scene, "cameras", [])
    for idx, cam in enumerate(cameras or []):
        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        camera_path = f"{root_path}/camera_{idx}"
        if rgb is not None:
            recording.log(
                f"{camera_path}/rgb",
                rr.Image(select_camera_env(rgb, env_index, unbatched_ndim=3)),
            )
        if depth is not None:
            selected_depth = select_camera_env(depth, env_index, unbatched_ndim=2)
            recording.log(
                f"{camera_path}/depth",
                rr.DepthImage(selected_depth),
            )
            _log_point_clouds_from_depth(
                recording,
                root_path,
                cam,
                selected_depth,
                camera_index=idx,
            )
