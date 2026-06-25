"""Transform logging is the backbone of the 3D view; verify paths and quaternions."""

from __future__ import annotations

import numpy as np

from genesis_rerun.sensors import transforms

from .conftest import FakeEntity, FakeLink, FakeScene


def test_quaternion_convention_wxyz_to_xyzw() -> None:
    # Genesis stores quaternions as (w, x, y, z); Rerun expects (x, y, z, w).
    wxyz = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(
        transforms.genesis_wxyz_to_rerun_xyzw(wxyz), [0.2, 0.3, 0.4, 0.1]
    )


def test_log_emits_one_transform_per_link_with_namespaced_paths(recording) -> None:
    scene = FakeScene(
        entities=[
            FakeEntity(
                "panda",
                links=[
                    FakeLink("link0", pos=[0, 0, 0], quat_wxyz=[1, 0, 0, 0]),
                    FakeLink("hand", pos=[0.3, 0.0, 0.5], quat_wxyz=[0, 1, 0, 0]),
                ],
            )
        ]
    )

    transforms.log(scene, recording, env_index=0, root_path="world")

    transform_entries = recording.archetypes("Transform3D")
    assert len(transform_entries) == 2
    assert recording.paths() == ["world/panda/link0", "world/panda/hand"]


def test_log_skips_links_without_pose_accessors(recording) -> None:
    class PoselessLink:
        name = "fixture"

    scene = FakeScene(entities=[FakeEntity("rig", links=[PoselessLink()])])

    transforms.log(scene, recording, env_index=0)

    assert recording.entries == []


def test_log_handles_batched_poses_without_crashing(recording) -> None:
    # A batched (n_envs, 3) pose must not raise; env selection itself is covered
    # by the select_env unit tests. Here we only assert it logs exactly one
    # transform for the chosen env.
    link = FakeLink("base", pos=[[0, 0, 0], [9, 9, 9]], quat_wxyz=[[1, 0, 0, 0], [1, 0, 0, 0]])
    scene = FakeScene(entities=[FakeEntity("rig", links=[link])])

    transforms.log(scene, recording, env_index=1)

    assert len(recording.archetypes("Transform3D")) == 1
