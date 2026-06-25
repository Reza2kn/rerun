"""Camera adapter: RGB image, depth image, and depth -> world point cloud."""

from __future__ import annotations

import numpy as np

from genesis_rerun.sensors import camera

from .conftest import FakeCamera, FakeScene


def _rgbd(height: int = 4, width: int = 4):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), 1.0, dtype=np.float32)  # all within (0.05, 6.0)
    return rgb, depth


def test_camera_logs_rgb_and_depth_images(recording) -> None:
    rgb, depth = _rgbd()
    scene = FakeScene(cameras=[FakeCamera(rgb, depth)])

    camera.log(scene, recording, env_index=0, root_path="world")

    assert recording.archetypes("Image"), "expected an RGB image"
    assert recording.archetypes("DepthImage"), "expected a depth image"
    assert "world/camera_0/rgb" in recording.paths()
    assert "world/camera_0/depth" in recording.paths()


def test_valid_depth_reprojects_into_a_world_point_cloud(recording) -> None:
    rgb, depth = _rgbd()
    scene = FakeScene(cameras=[FakeCamera(rgb, depth, pos=(0, 0, 1), lookat=(0, 0, 0))])

    camera.log(scene, recording, env_index=0, root_path="world")

    point_clouds = recording.archetypes("Points3D")
    assert point_clouds, "valid finite depth should produce at least one Points3D"
    assert any("point_cloud" in e.path for e in point_clouds)


def test_point_clouds_only_built_for_the_first_camera(recording) -> None:
    rgb, depth = _rgbd()
    scene = FakeScene(cameras=[FakeCamera(rgb, depth), FakeCamera(rgb, depth)])

    camera.log(scene, recording, env_index=0, root_path="world")

    # Both cameras log rgb+depth, but only camera_0 emits point clouds.
    assert all("camera_1" not in e.path for e in recording.archetypes("Points3D"))
