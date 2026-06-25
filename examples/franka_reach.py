"""Franka tabletop pick-and-place demo streamed to Rerun.

A richer companion to ``minimal_genesis_rerun.py``: a Franka Panda sweeps an
orange toward a bowl while every link transform, both camera RGB/D streams,
contact-force arrows, and IMU scalars are logged through ``GenesisRerunLogger``
into a multi-view Rerun blueprint. Runs with no external assets (a primitive
orange and counter are built in code); set ``LIGHTWHEEL_KITCHEN`` to a USD stage
for a textured backdrop.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from genesis_rerun import GenesisRerunLogger

# Optional photoreal backdrop. Absent by default -> the demo builds a primitive
# kitchen instead, so it runs with no external assets. Point this at a USD stage
# to use a textured room (e.g. export LIGHTWHEEL_KITCHEN=/path/to/KitchenRoom.usda).
_DEFAULT_KITCHEN = "assets/lightwheel/KitchenRoom_visual.usda"
LIGHTWHEEL_KITCHEN_PATH = Path(os.environ.get("LIGHTWHEEL_KITCHEN", _DEFAULT_KITCHEN))
HOME_QPOS = np.array([0.0, -0.55, 0.0, -2.35, 0.0, 1.85, 0.78, 0.04, 0.04], dtype=float)
HOME_HAND_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
ARM_DOFS = list(range(7))
GRIPPER_DOFS = [7, 8]

# Measured from assets/lightwheel/KitchenRoom_visual.usda:
# /root/Kitchen_InsularShelf_01/Kitchen_InsularShelf/Kitchen_InsularShelf
# has bounds center ~= (0.213, 0.225, 0.429), size ~= (1.150, 0.761, 0.858).
TABLE_CENTER = np.array([0.213, 0.225], dtype=float)
TABLE_SIZE = np.array([1.15, 0.761], dtype=float)
TABLE_TOP_Z = 0.858
ORANGE_HALF_HEIGHT = 0.020
ORANGE_RADIUS = 0.038
ORANGE_START = np.array([0.22, -0.03, TABLE_TOP_Z + ORANGE_HALF_HEIGHT + 0.012], dtype=float)
BOWL_CENTER = np.array([0.035, -0.065, TABLE_TOP_Z + 0.035], dtype=float)
APPROACH_CLEARANCE = 0.28
PUSH_CLEARANCE = 0.070
ROBOT_BASE_POS = (0.48, -0.62, 0.395)

WORLD_CAMERA_PATH = "world/camera_0"
WRIST_CAMERA_PATH = "world/camera_1"


def as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def lerp(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    amount = np.clip(amount, 0.0, 1.0)
    eased = amount * amount * (3.0 - 2.0 * amount)
    return (1.0 - eased) * start + eased * end


def wrist_camera_offset() -> np.ndarray:
    pos = np.array([-0.16, 0.0, 0.08], dtype=np.float32)
    lookat = np.array([0.36, 0.0, 0.00], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return as_numpy(gu.pos_lookat_up_to_T(pos, lookat, up))


def update_wrist_camera(camera: Any) -> None:
    if hasattr(camera, "move_to_attach"):
        camera.move_to_attach()


def add_box(
    scene: Any,
    *,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float, float],
    fixed: bool = True,
    collision: bool = True,
    visualization: bool = True,
) -> Any:
    return scene.add_entity(
        gs.morphs.Box(
            pos=pos, size=size, fixed=fixed, collision=collision, visualization=visualization
        ),
        surface=gs.surfaces.Rough(color=color),
        material=gs.materials.Rigid(friction=1.0),
        name=name,
    )


def add_cylinder(
    scene: Any,
    *,
    name: str,
    pos: tuple[float, float, float],
    radius: float,
    height: float,
    color: tuple[float, float, float, float],
    fixed: bool = True,
    collision: bool = True,
    visualization: bool = True,
) -> Any:
    return scene.add_entity(
        gs.morphs.Cylinder(
            pos=pos,
            radius=radius,
            height=height,
            fixed=fixed,
            collision=collision,
            visualization=visualization,
        ),
        surface=gs.surfaces.Rough(color=color),
        material=gs.materials.Rigid(friction=1.0),
        name=name,
    )


def add_orange(scene: Any) -> Any:
    """A movable primitive 'orange' -- no external mesh required."""

    return scene.add_entity(
        gs.morphs.Sphere(
            pos=tuple(float(x) for x in ORANGE_START),
            radius=ORANGE_RADIUS,
            fixed=False,
            collision=True,
        ),
        surface=gs.surfaces.Rough(color=(0.95, 0.55, 0.12, 1.0)),
        material=gs.materials.Rigid(rho=280.0, friction=1.1, coup_restitution=0.12),
        name="orange",
    )


def add_lightwheel_kitchen(scene: Any) -> bool:
    if not LIGHTWHEEL_KITCHEN_PATH.exists():
        return False
    scene.add_entity(
        gs.morphs.USD(
            file=str(LIGHTWHEEL_KITCHEN_PATH),
            fixed=True,
            collision=False,
            scale=1.0,
            decimate=True,
            decimate_face_num=2200,
        ),
        name="lightwheel_kitchen_visual",
    )
    return True


def build_tabletop_scene(scene: Any, *, lightwheel_loaded: bool) -> Any:
    if not lightwheel_loaded:
        scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Rough(color=(0.34, 0.36, 0.32, 1.0)),
            name="kitchen_floor",
        )
    add_box(
        scene,
        name="task_counter_collision" if lightwheel_loaded else "counter_table",
        pos=(float(TABLE_CENTER[0]), float(TABLE_CENTER[1]), TABLE_TOP_Z - 0.04),
        size=(float(TABLE_SIZE[0]), float(TABLE_SIZE[1]), 0.08),
        color=(0.62, 0.50, 0.37, 1.0),
        visualization=not lightwheel_loaded,
    )
    if lightwheel_loaded:
        add_box(
            scene,
            name="task_backstop_collision",
            pos=(
                float(TABLE_CENTER[0]),
                float(TABLE_CENTER[1] + TABLE_SIZE[1] / 2.0),
                TABLE_TOP_Z + 0.22,
            ),
            size=(float(TABLE_SIZE[0]), 0.05, 0.44),
            color=(0.42, 0.50, 0.56, 1.0),
            visualization=False,
        )
        add_box(
            scene,
            name="task_left_guard_collision",
            pos=(
                float(TABLE_CENTER[0] + TABLE_SIZE[0] / 2.0),
                float(TABLE_CENTER[1]),
                TABLE_TOP_Z + 0.22,
            ),
            size=(0.05, float(TABLE_SIZE[1]), 0.44),
            color=(0.42, 0.50, 0.56, 1.0),
            visualization=False,
        )
        add_box(
            scene,
            name="task_right_guard_collision",
            pos=(
                float(TABLE_CENTER[0] - TABLE_SIZE[0] / 2.0),
                float(TABLE_CENTER[1]),
                TABLE_TOP_Z + 0.22,
            ),
            size=(0.05, float(TABLE_SIZE[1]), 0.44),
            color=(0.42, 0.50, 0.56, 1.0),
            visualization=False,
        )
    if not lightwheel_loaded:
        add_box(
            scene,
            name="backsplash_wall",
            pos=(0.48, 0.42, TABLE_TOP_Z + 0.26),
            size=(1.12, 0.045, 0.52),
            color=(0.42, 0.50, 0.56, 1.0),
        )
        add_box(
            scene,
            name="left_cabinet",
            pos=(0.04, 0.47, TABLE_TOP_Z + 0.57),
            size=(0.36, 0.12, 0.28),
            color=(0.30, 0.38, 0.45, 1.0),
        )
        add_box(
            scene,
            name="right_cabinet",
            pos=(0.78, 0.47, TABLE_TOP_Z + 0.57),
            size=(0.36, 0.12, 0.28),
            color=(0.30, 0.38, 0.45, 1.0),
        )

    bowl_color = (0.92, 0.95, 0.98, 1.0)
    add_cylinder(
        scene,
        name="bowl_visual",
        pos=(float(BOWL_CENTER[0]), float(BOWL_CENTER[1]), TABLE_TOP_Z + 0.018),
        radius=0.13,
        height=0.035,
        color=bowl_color,
        collision=False,
    )
    return add_orange(scene)


def plan_pose(step: int, total_steps: int) -> tuple[np.ndarray, float, str]:
    # Physical shove/yeet path: approach from behind the slice, lower above
    # the counter, close the gripper, sweep through the orange toward the bowl,
    # then open and retreat. The orange itself is not repositioned or velocity
    # injected after scene initialization.
    fingertip_offset_comp = np.array([0.02, 0.02, 0.0], dtype=float)
    above_orange = (
        ORANGE_START
        + fingertip_offset_comp
        + np.array([0.0, -0.18, APPROACH_CLEARANCE], dtype=float)
    )
    push_start = (
        ORANGE_START + fingertip_offset_comp + np.array([0.0, -0.14, PUSH_CLEARANCE], dtype=float)
    )
    push_contact = (
        ORANGE_START + fingertip_offset_comp + np.array([0.0, 0.03, PUSH_CLEARANCE], dtype=float)
    )
    push_release = np.array([0.14, 0.13, ORANGE_START[2] + PUSH_CLEARANCE], dtype=float)
    retreat = np.array([0.30, -0.18, TABLE_TOP_Z + 0.48], dtype=float)

    phase = step / max(1, total_steps - 1)
    if phase < 0.20:
        return lerp(retreat, above_orange, phase / 0.20), 0.060, "approach"
    if phase < 0.40:
        return lerp(above_orange, push_start, (phase - 0.20) / 0.20), 0.060, "descend"
    if phase < 0.58:
        return lerp(push_start, push_contact, (phase - 0.40) / 0.18), 0.026, "contact"
    if phase < 0.82:
        return lerp(push_contact, push_release, (phase - 0.58) / 0.24), 0.026, "sweep"
    if phase < 0.90:
        return push_release + np.array([0.0, 0.02, 0.03], dtype=float), 0.060, "open_release"
    return (
        lerp(
            push_release + np.array([0.0, 0.02, 0.03], dtype=float), retreat, (phase - 0.90) / 0.10
        ),
        0.060,
        "retreat",
    )


def control_pick_place(
    robot: object, hand_link: object, target_pos: np.ndarray, grip: float
) -> np.ndarray:
    qpos = robot.inverse_kinematics(
        hand_link,
        pos=target_pos,
        quat=HOME_HAND_QUAT,
        init_qpos=robot.get_qpos(),
        rot_mask=[True, True, True],
        dofs_idx_local=ARM_DOFS,
        max_solver_iters=36,
    )
    qpos_np = as_numpy(qpos).astype(float)
    command = np.array(HOME_QPOS, copy=True)
    command[:7] = qpos_np[:7]
    command[GRIPPER_DOFS] = grip
    robot.control_dofs_position(command)
    return command


def entity_pos(entity: Any) -> np.ndarray:
    if hasattr(entity, "get_pos"):
        return as_numpy(entity.get_pos()).reshape(-1)[:3]
    return np.array([math.nan, math.nan, math.nan], dtype=float)


def log_task_state(
    step: int,
    robot: object,
    hand_link: object,
    orange: Any,
    target_pos: np.ndarray,
    grip: float,
    phase_name: str,
) -> None:
    hand_pos = as_numpy(hand_link.get_pos()).reshape(-1)[:3]
    orange_pos = entity_pos(orange)
    orange_finite = np.all(np.isfinite(orange_pos))
    bowl_error = (
        float(np.linalg.norm(orange_pos[:2] - BOWL_CENTER[:2])) if orange_finite else math.nan
    )
    rr.log(
        "world/task/hand_target_error_m", rr.Scalars(float(np.linalg.norm(hand_pos - target_pos)))
    )
    rr.log("world/task/orange_to_bowl_xy_m", rr.Scalars(bowl_error))
    rr.log("world/task/gripper_command_m", rr.Scalars(float(grip)))
    rr.log("world/task/phase_index", rr.Scalars(float(step)))
    qpos = as_numpy(robot.get_qpos())
    qvel = as_numpy(robot.get_dofs_velocity())
    for idx, (position, velocity) in enumerate(zip(qpos, qvel, strict=False)):
        rr.log(f"world/joints/joint_{idx}/position", rr.Scalars(float(position)))
        rr.log(f"world/joints/joint_{idx}/velocity", rr.Scalars(float(velocity)))
    rr.log("world/task/phase", rr.TextDocument(phase_name))


def build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                name="Genesis kitchen orange throw",
                contents=[
                    "world/panda/**",
                    "world/lightwheel_kitchen_visual_usd/**",
                    "world/orange/**",
                    "world/bowl_*/**",
                    "world/counter_table/**",
                    "world/backsplash_wall/**",
                    "world/*cabinet/**",
                    "world/kitchen_floor/**",
                    "-world/task_*_collision/**",
                    "-world/camera_*/**",
                    "-world/joints/**",
                    "-world/imu_*/**",
                    "-world/*point_cloud/**",
                    "-world/task/**",
                ],
                line_grid=True,
            ),
            rrb.Vertical(
                rrb.Grid(
                    rrb.Spatial2DView(origin=f"{WORLD_CAMERA_PATH}/rgb", name="World RGB"),
                    rrb.Spatial2DView(origin=f"{WRIST_CAMERA_PATH}/rgb", name="Wrist RGB"),
                    rrb.Spatial2DView(origin=f"{WORLD_CAMERA_PATH}/depth", name="World depth"),
                    rrb.Spatial2DView(origin=f"{WRIST_CAMERA_PATH}/depth", name="Wrist depth"),
                    grid_columns=2,
                ),
                rrb.Grid(
                    rrb.Spatial3DView(
                        origin="world",
                        name="Lidar point cloud",
                        contents=["world/lidar_point_cloud/points"],
                        line_grid=True,
                    ),
                    rrb.Spatial3DView(
                        origin="world",
                        name="Depth point cloud",
                        contents=["world/depth_point_cloud/points"],
                        line_grid=True,
                    ),
                    grid_columns=2,
                ),
                row_shares=[3, 2],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="sim_step", expanded=True),
        rrb.SelectionPanel(state="collapsed"),
        auto_layout=False,
    )


def build_render_verification_blueprint() -> rrb.Blueprint:
    """Single-view native layout for screenshot verification."""

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="world",
            name="Textured Genesis kitchen",
            contents=[
                "world/panda/**",
                "world/lightwheel_kitchen_visual_usd/**",
                "world/orange_slice/**",
                "world/bowl_*/**",
                "-world/task_*_collision/**",
                "-world/camera_*/**",
                "-world/joints/**",
                "-world/imu_*/**",
                "-world/*point_cloud/**",
                "-world/task/**",
            ],
            line_grid=True,
        ),
        rrb.TimePanel(timeline="sim_step", expanded=False),
        rrb.SelectionPanel(state="collapsed"),
        auto_layout=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--recording-id", default="genesis-rerun-franka-pick-place")
    parser.add_argument("--no-spawn", action="store_true", help="Do not open a Rerun viewer.")
    parser.add_argument(
        "--save-rrd",
        type=Path,
        default=None,
        help="Write the Rerun recording to this .rrd file after the run.",
    )
    parser.add_argument(
        "--render-verification-blueprint",
        action="store_true",
        help="Use a compact native-viewer layout for screenshot verification.",
    )
    args = parser.parse_args()

    gs.init()
    blueprint = (
        build_render_verification_blueprint()
        if args.render_verification_blueprint
        else build_blueprint()
    )
    rr.init(args.recording_id, spawn=not args.no_spawn, default_blueprint=blueprint)
    if args.save_rrd is not None:
        args.save_rrd.parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save_rrd, default_blueprint=blueprint)

    scene = gs.Scene(show_viewer=False)
    lightwheel_loaded = add_lightwheel_kitchen(scene)
    orange = build_tabletop_scene(scene, lightwheel_loaded=lightwheel_loaded)
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=ROBOT_BASE_POS), name="panda"
    )
    scene.add_camera(
        res=(640, 480),
        pos=(1.35, -1.18, 1.42),
        lookat=(0.24, 0.16, 0.92),
        fov=55,
    )
    wrist_camera = scene.add_camera(
        res=(640, 480),
        pos=(0.42, -0.18, 1.16),
        lookat=tuple(float(x) for x in ORANGE_START),
        fov=70,
    )

    hand_link = robot.links[8]
    sensor_kwargs = {
        "entity_idx": robot.idx,
        "link_idx_local": hand_link.idx_local,
    }
    sensor_handles = [
        scene.add_sensor(gs.sensors.ContactForce(**sensor_kwargs)),
        scene.add_sensor(gs.sensors.IMU(**sensor_kwargs)),
    ]
    scene.build()
    robot.set_qpos(HOME_QPOS)
    robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2500, 2000, 2000, 320, 320]))
    robot.set_dofs_kv(np.array([450, 450, 350, 350, 250, 200, 200, 32, 32]))
    wrist_camera.attach(hand_link, wrist_camera_offset())

    logger = GenesisRerunLogger(
        scene,
        sensors=["transforms", "camera", "contact", "imu"],
        sensor_handles=sensor_handles,
    )
    logger.log_setup()

    dt = getattr(getattr(scene, "sim_options", None), "dt", 1.0 / 60.0)
    for step in range(args.steps):
        target_pos, grip, phase_name = plan_pose(step, args.steps)
        control_pick_place(robot, hand_link, target_pos, grip)
        scene.step()
        update_wrist_camera(wrist_camera)
        logger.log_step(step, step * float(dt))
        log_task_state(step, robot, hand_link, orange, target_pos, grip, phase_name)


if __name__ == "__main__":
    main()
