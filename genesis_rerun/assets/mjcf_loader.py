"""Visual mesh extraction helpers for Genesis rigid entities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr

from genesis_rerun.assets.usd_loader import log_usd_meshes


@dataclass(frozen=True, slots=True)
class LinkMesh:
    """A renderable mesh associated with one Genesis link."""

    entity_path: str
    vertex_positions: np.ndarray
    triangle_indices: np.ndarray
    vertex_colors: np.ndarray


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(-1)
    if quat.size != 4:
        return np.eye(3)
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return np.eye(3)
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _geom_vertices_in_link_frame(geom: Any) -> np.ndarray:
    trimesh = geom.get_trimesh()
    vertices = np.asarray(trimesh.vertices, dtype=float)
    init_pos = getattr(geom, "init_pos", None)
    init_quat = getattr(geom, "init_quat", None)
    if init_pos is None or init_quat is None:
        return vertices
    rotation = _quat_wxyz_to_matrix(np.asarray(init_quat, dtype=float))
    return vertices @ rotation.T + np.asarray(init_pos, dtype=float).reshape(1, 3)


def iter_link_meshes(entity: Any, *, root_path: str = "world") -> list[LinkMesh]:
    """Extract trimesh-backed visual/collision geometry from a Genesis entity.

    Genesis' exact visual-geometry API has changed across releases, so this helper
    intentionally accepts any geometry object that exposes `get_trimesh()`.
    """

    meshes: list[LinkMesh] = []
    entity_name = getattr(entity, "name", f"entity_{getattr(entity, 'idx', 0)}")
    for link in getattr(entity, "links", []):
        raw_name = getattr(link, "name", f"link_{getattr(link, 'idx_local', 0)}")
        link_name = str(raw_name).strip("/") or "root"
        geoms = list(getattr(link, "vgeoms", []) or [])
        if not geoms:
            geoms = list(getattr(link, "geoms", []) or [])
        vertices: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        vertex_offset = 0
        for geom_index, geom in enumerate(geoms):
            if not hasattr(geom, "get_trimesh"):
                continue
            trimesh = geom.get_trimesh()
            semantic_name = f"{entity_name}/{link_name}".lower()
            if "ball" in semantic_name:
                color = (220, 32, 28, 255)
            elif "bowl" in semantic_name:
                color = (245, 248, 252, 255)
            elif "counter" in semantic_name or "table" in semantic_name:
                color = (154, 128, 96, 255)
            elif any(token in semantic_name for token in ("cabinet", "backsplash", "wall")):
                color = (96, 116, 132, 255)
            elif "floor" in semantic_name or "plane" in semantic_name:
                color = (84, 88, 82, 255)
            elif "finger" in link_name or "hand" in link_name:
                color = (35, 38, 42, 255)
            elif geom_index % 2:
                color = (45, 48, 54, 255)
            else:
                color = (215, 218, 220, 255)
            geom_vertices = _geom_vertices_in_link_frame(geom)
            geom_faces = np.asarray(trimesh.faces, dtype=np.uint32)
            vertices.append(geom_vertices)
            faces.append(geom_faces + vertex_offset)
            colors.append(np.tile(np.asarray(color, dtype=np.uint8), (len(geom_vertices), 1)))
            vertex_offset += len(geom_vertices)
        if vertices and faces:
            meshes.append(
                LinkMesh(
                    entity_path=f"{root_path}/{entity_name}/{link_name}",
                    vertex_positions=np.concatenate(vertices, axis=0),
                    triangle_indices=np.concatenate(faces, axis=0),
                    vertex_colors=np.concatenate(colors, axis=0),
                )
            )
    return meshes


def log_entity_meshes(entity: Any, recording: Any = rr, *, root_path: str = "world") -> None:
    """Log all available meshes for a Genesis entity once."""

    morph_file = getattr(getattr(entity, "morph", None), "file", None)
    entity_name = getattr(entity, "name", "")
    if morph_file is not None and str(morph_file).lower().endswith((".usd", ".usda", ".usdc")):
        if entity_name != "lightwheel_kitchen_visual":
            link = next(iter(getattr(entity, "links", []) or []), None)
            link_name = str(getattr(link, "name", "root")).strip("/") or "root"
            log_usd_meshes(
                Path(str(morph_file)),
                recording,
                root_path=f"{root_path}/{entity_name}/{link_name}/visual_usd",
            )
            return
        log_usd_meshes(
            Path(str(morph_file)),
            recording,
            root_path=f"{root_path}/{entity_name}_usd",
        )
        return

    for mesh in iter_link_meshes(entity, root_path=root_path):
        recording.log(
            mesh.entity_path,
            rr.Mesh3D(
                vertex_positions=mesh.vertex_positions,
                triangle_indices=mesh.triangle_indices,
                vertex_colors=mesh.vertex_colors,
            ),
            static=True,
        )
