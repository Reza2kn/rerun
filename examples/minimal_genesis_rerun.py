"""Minimal, self-contained Genesis -> Rerun example.

Builds a tiny tabletop scene -- a ground plane, a falling box, and one camera --
and streams it to Rerun through :class:`GenesisRerunLogger`. There are no external
assets and no hard-coded paths, so it runs anywhere Genesis itself runs.

    # stream live into a spawned Rerun viewer
    python examples/minimal_genesis_rerun.py

    # or write a recording you can open later (or share)
    python examples/minimal_genesis_rerun.py --no-spawn --save-rrd minimal.rrd

What it demonstrates:
  * the public bridge entry point (`GenesisRerunLogger`) wired to a real scene,
  * static mesh + ViewCoordinates setup via `log_setup()`,
  * per-step link transforms and camera RGB/D streamed under `sim_step`/`sim_time`,
  * a hand-authored Rerun blueprint that lays out a 3D view next to the camera.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import genesis as gs
import rerun as rr
import rerun.blueprint as rrb

from genesis_rerun import GenesisRerunLogger


def build_blueprint() -> rrb.Blueprint:
    """A 3D scene view beside the camera's RGB and depth panels."""

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="Genesis scene"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/camera_0/rgb", name="camera RGB"),
                rrb.Spatial2DView(origin="world/camera_0/depth", name="camera depth"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=240, help="number of sim steps to log")
    parser.add_argument("--no-spawn", action="store_true", help="do not open a Rerun viewer")
    parser.add_argument(
        "--save-rrd", type=Path, default=None, help="also write the recording to this .rrd file"
    )
    args = parser.parse_args()

    blueprint = build_blueprint()
    rr.init("genesis_rerun_minimal", spawn=not args.no_spawn, default_blueprint=blueprint)
    if args.save_rrd is not None:
        args.save_rrd.parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save_rrd, default_blueprint=blueprint)

    gs.init()
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane(), name="ground")
    scene.add_entity(
        gs.morphs.Box(pos=(0.0, 0.0, 0.4), size=(0.1, 0.1, 0.1)),
        surface=gs.surfaces.Rough(color=(0.85, 0.4, 0.2, 1.0)),
        name="box",
    )
    scene.add_camera(res=(320, 240), pos=(1.2, -1.2, 0.9), lookat=(0.0, 0.0, 0.2), fov=50)
    scene.build()

    # The bridge is the only Rerun-facing code below: setup once, then one call
    # per step. `sensors` selects which adapters run; here, link poses + camera.
    logger = GenesisRerunLogger(scene, sensors=["transforms", "camera"])
    logger.log_setup()

    dt = float(getattr(getattr(scene, "sim_options", None), "dt", 1.0 / 60.0))
    for step in range(args.steps):
        scene.step()
        logger.log_step(step, step * dt)


if __name__ == "__main__":
    main()
