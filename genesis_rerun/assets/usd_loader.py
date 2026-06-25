"""USD visual mesh logging helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr

MAX_TEXTURE_SIZE = 1024


@dataclass(frozen=True, slots=True)
class UsdMesh:
    """A static mesh extracted from a USD stage."""

    entity_path: str
    vertex_positions: np.ndarray
    triangle_indices: np.ndarray
    vertex_colors: np.ndarray | None = None
    vertex_texcoords: np.ndarray | None = None
    albedo_texture: np.ndarray | None = None
    albedo_factor: tuple[float, float, float, float] | None = None


def _safe_name(path: str) -> str:
    return path.strip("/").replace("/", "_").replace(":", "_") or "root"


def _triangulate(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in face_counts.astype(int):
        face = face_indices[offset : offset + count].astype(int)
        offset += count
        if count < 3:
            continue
        for idx in range(1, count - 1):
            triangles.append((int(face[0]), int(face[idx]), int(face[idx + 1])))
    return np.asarray(triangles, dtype=np.uint32)


def _triangulate_faces(
    face_counts: np.ndarray, face_indices: np.ndarray, face_ids: np.ndarray
) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    offsets = np.concatenate([[0], np.cumsum(face_counts[:-1])]).astype(np.int64)
    for face_id in face_ids.astype(int):
        count = int(face_counts[face_id])
        if count < 3:
            continue
        start = int(offsets[face_id])
        face = face_indices[start : start + count].astype(int)
        for idx in range(1, count - 1):
            triangles.append((int(face[0]), int(face[idx]), int(face[idx + 1])))
    return np.asarray(triangles, dtype=np.uint32)


def _semantic_color(path: str) -> tuple[int, int, int, int]:
    lowered = path.lower()
    if "floor" in lowered or "ground" in lowered:
        return (95, 98, 90, 255)
    if "cabinet" in lowered or "shelf" in lowered:
        return (132, 110, 84, 255)
    if "refrigerator" in lowered or "dishwasher" in lowered or "oven" in lowered:
        return (165, 170, 176, 255)
    if "sink" in lowered or "stove" in lowered or "microwave" in lowered:
        return (90, 98, 108, 255)
    if "orange" in lowered:
        return (230, 112, 30, 255)
    if "bottle" in lowered or "pot" in lowered:
        return (70, 110, 130, 255)
    return (178, 176, 166, 255)


def _asset_path_to_path(asset: Any, usd_dir: Path) -> Path | None:
    if asset is None:
        return None
    raw = getattr(asset, "resolvedPath", "") or getattr(asset, "path", "") or str(asset)
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = usd_dir / path
    return path if path.exists() else None


def _load_texture(path: Path, cache: dict[Path, np.ndarray]) -> np.ndarray | None:
    cached = cache.get(path)
    if cached is not None:
        return cached
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(path) as image:
            image = image.convert("RGBA")
            image.thumbnail((MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE))
            texture = np.asarray(image, dtype=np.uint8).copy()
    except Exception:
        return None

    cache[path] = texture
    return texture


def _find_texture_path(connectable: Any, usd_dir: Path) -> Path | None:
    from pxr import UsdShade

    prim = connectable.GetPrim() if hasattr(connectable, "GetPrim") else connectable
    stack = [prim]
    while stack:
        item = stack.pop()
        shader = UsdShade.Shader(item)
        file_input = shader.GetInput("file") if shader else None
        if file_input is not None:
            path = _asset_path_to_path(file_input.Get(), usd_dir)
            if path is not None:
                return path
        stack.extend(list(item.GetAllChildren()))
    return None


def _material_info(
    material: Any, usd_dir: Path, texture_cache: dict[Path, np.ndarray]
) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None]:

    if not material:
        return None, None
    surface = material.ComputeSurfaceSource()[0]
    if not surface:
        # Omniverse/Lightwheel materials are often MDL-only (no
        # UsdPreviewSurface); the MDL shader carries direct asset inputs.
        surface = material.ComputeSurfaceSource("mdl")[0]
    if not surface:
        return None, None

    # UsdPreviewSurface style: diffuseColor connected to a UsdUVTexture.
    # NOTE: GetInput on a missing name returns an INVALID (falsy, non-None)
    # UsdShadeInput; calling GetConnectedSources on it aborts in C++.
    diffuse = surface.GetInput("diffuseColor")
    if diffuse:
        sources = diffuse.GetConnectedSources()[0]
        for source in sources:
            texture_path = _find_texture_path(source.source, usd_dir)
            if texture_path is not None:
                texture = _load_texture(texture_path, texture_cache)
                if texture is not None:
                    return texture, None
        value = diffuse.Get()
        if value is not None:
            rgb = tuple(float(v) for v in value[:3])
            return None, (rgb[0], rgb[1], rgb[2], 1.0)

    # MDL (OmniPBR-style): texture file is a direct asset input.
    for name in ("diffuse_texture", "albedo_map", "base_color_texture"):
        inp = surface.GetInput(name)
        if not inp:
            continue
        texture_path = _asset_path_to_path(inp.Get(), usd_dir)
        if texture_path is not None:
            texture = _load_texture(texture_path, texture_cache)
            if texture is not None:
                return texture, None

    # MDL constant tint fallback.
    for name in ("diffuse_color_constant", "diffuse_tint", "base_color_constant"):
        inp = surface.GetInput(name)
        value = inp.Get() if inp else None
        if value is not None:
            rgb = tuple(float(v) for v in value[:3])
            return None, (rgb[0], rgb[1], rgb[2], 1.0)
    return None, None


def _get_uv_data(mesh: Any) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    from pxr import UsdGeom

    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st")
    if not primvar:
        return None, None, None
    values = primvar.Get()
    if values is None:
        return None, None, None
    indices = primvar.GetIndices()
    uv_values = np.asarray(values, dtype=np.float32)
    has_indices = indices is not None and len(indices)
    uv_indices = np.asarray(indices, dtype=np.int64) if has_indices else None
    return uv_values, uv_indices, primvar.GetInterpolation()


def _resolve_uv(
    *,
    uv_values: np.ndarray | None,
    uv_indices: np.ndarray | None,
    interpolation: str | None,
    face_id: int,
    face_vertex_offset: int,
    vertex_index: int,
) -> np.ndarray | None:
    if uv_values is None or interpolation is None:
        return None
    if interpolation == "faceVarying":
        uv_id = face_vertex_offset
    elif interpolation in {"vertex", "varying"}:
        uv_id = vertex_index
    elif interpolation == "uniform":
        uv_id = face_id
    elif interpolation == "constant":
        uv_id = 0
    else:
        return None
    if uv_indices is not None:
        if uv_id >= len(uv_indices):
            return None
        uv_id = int(uv_indices[uv_id])
    if uv_id >= len(uv_values):
        return None
    uv = np.asarray(uv_values[uv_id], dtype=np.float32).copy()
    uv[1] = 1.0 - uv[1]
    return uv


def _expand_textured_subset(
    *,
    world_vertices: np.ndarray,
    face_counts: np.ndarray,
    face_indices: np.ndarray,
    face_ids: np.ndarray,
    uv_values: np.ndarray | None,
    uv_indices: np.ndarray | None,
    uv_interpolation: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    positions: list[np.ndarray] = []
    texcoords: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    offsets = np.concatenate([[0], np.cumsum(face_counts[:-1])]).astype(np.int64)

    for face_id in face_ids.astype(int):
        count = int(face_counts[face_id])
        if count < 3:
            continue
        start = int(offsets[face_id])
        face = face_indices[start : start + count].astype(int)
        for idx in range(1, count - 1):
            tri = (0, idx, idx + 1)
            tri_indices: list[int] = []
            for corner in tri:
                vertex_index = int(face[corner])
                positions.append(world_vertices[vertex_index])
                uv = _resolve_uv(
                    uv_values=uv_values,
                    uv_indices=uv_indices,
                    interpolation=uv_interpolation,
                    face_id=face_id,
                    face_vertex_offset=start + corner,
                    vertex_index=vertex_index,
                )
                if uv is not None:
                    texcoords.append(uv)
                tri_indices.append(len(positions) - 1)
            triangles.append(tuple(tri_indices))

    if not triangles:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32), None
    uv_array = np.asarray(texcoords, dtype=np.float32) if len(texcoords) == len(positions) else None
    return np.asarray(positions, dtype=np.float32), np.asarray(triangles, dtype=np.uint32), uv_array


def _mesh_subsets(prim: Any, face_count: int) -> list[tuple[str, np.ndarray, Any]]:
    from pxr import UsdGeom, UsdShade

    subsets: list[tuple[str, np.ndarray, Any]] = []
    covered: set[int] = set()
    for child in prim.GetChildren():
        if not child.IsA(UsdGeom.Subset):
            continue
        subset = UsdGeom.Subset(child)
        indices = subset.GetIndicesAttr().Get()
        if indices is None or len(indices) == 0:
            continue
        face_ids = np.asarray(indices, dtype=np.int64)
        covered.update(int(idx) for idx in face_ids)
        material = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial()[0]
        subsets.append((_safe_name(str(child.GetName())), face_ids, material))

    missing = np.asarray([idx for idx in range(face_count) if idx not in covered], dtype=np.int64)
    if len(missing):
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        subsets.append(("default", missing, material))
    if not subsets:
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        subsets.append(("default", np.arange(face_count, dtype=np.int64), material))
    return subsets


def iter_usd_meshes(
    usd_path: Path, *, root_path: str = "world/lightwheel_kitchen_usd"
) -> list[UsdMesh]:
    """Extract renderable meshes from a USD file.

    This intentionally logs vertices already transformed to world space. Rerun then
    receives stable static mesh archetypes even when Genesis collapses the imported
    USD into one coarse rigid entity for physics/visualization.
    """

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return []

    cache = UsdGeom.XformCache()
    texture_cache: dict[Path, np.ndarray] = {}
    meshes: list[UsdMesh] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr().Get()
        counts_attr = mesh.GetFaceVertexCountsAttr().Get()
        indices_attr = mesh.GetFaceVertexIndicesAttr().Get()
        if points_attr is None or counts_attr is None or indices_attr is None:
            continue

        vertices = np.asarray(points_attr, dtype=np.float64)
        if vertices.size == 0:
            continue
        face_counts = np.asarray(counts_attr, dtype=np.int64)
        face_indices = np.asarray(indices_attr, dtype=np.int64)
        local_to_world = cache.GetLocalToWorldTransform(prim)
        transform = np.asarray(local_to_world, dtype=np.float64).reshape(4, 4).T
        hom = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
        world_vertices = (hom @ transform.T)[:, :3].astype(np.float32)

        uv_values, uv_indices, uv_interpolation = _get_uv_data(mesh)
        for subset_name, face_ids, material in _mesh_subsets(prim, len(face_counts)):
            texture, albedo_factor = _material_info(material, usd_path.parent, texture_cache)
            positions, triangles, texcoords = _expand_textured_subset(
                world_vertices=world_vertices,
                face_counts=face_counts,
                face_indices=face_indices,
                face_ids=face_ids,
                uv_values=uv_values,
                uv_indices=uv_indices,
                uv_interpolation=uv_interpolation,
            )
            if triangles.size == 0:
                continue

            if texture is not None and texcoords is None and len(positions):
                # MDL world-aligned textures (walls/floors) ship no UVs.
                # Samplers clamp out-of-range UVs (meter-scale UVs render as
                # one edge texel), so bake the repetition INTO the image:
                # tile it ~1m per period and normalize UVs to [0, 1].
                lo3 = positions.min(axis=0)
                ext = positions.max(axis=0) - lo3
                axes = list(np.argsort(ext))[1:]
                span_u = max(float(ext[axes[0]]), 1e-6)
                span_v = max(float(ext[axes[1]]), 1e-6)
                rep_u = max(1, min(8, int(np.ceil(span_u))))
                rep_v = max(1, min(8, int(np.ceil(span_v))))
                tiled = np.tile(texture, (rep_v, rep_u, 1))
                if max(tiled.shape[:2]) > 2048:
                    from PIL import Image as _Image

                    img = _Image.fromarray(tiled)
                    img.thumbnail((2048, 2048))
                    tiled = np.asarray(img, dtype=np.uint8).copy()
                texture = tiled
                u = (positions[:, axes[0]] - lo3[axes[0]]) / span_u
                v = (positions[:, axes[1]] - lo3[axes[1]]) / span_v
                texcoords = np.stack([u, v], axis=1).astype(np.float32)

            vertex_colors = None
            if texture is None and albedo_factor is None:
                color = np.asarray(_semantic_color(str(prim.GetPath())), dtype=np.uint8)
                vertex_colors = np.tile(color, (len(positions), 1))

            suffix = "" if subset_name == "default" else f"_{subset_name}"
            meshes.append(
                UsdMesh(
                    entity_path=f"{root_path}/{_safe_name(str(prim.GetPath()))}{suffix}",
                    vertex_positions=positions,
                    triangle_indices=triangles,
                    vertex_colors=vertex_colors,
                    vertex_texcoords=texcoords if texture is not None else None,
                    albedo_texture=texture if texcoords is not None else None,
                    albedo_factor=albedo_factor,
                )
            )
    return meshes


def log_usd_meshes(
    usd_path: Path, recording: Any = rr, *, root_path: str = "world/lightwheel_kitchen_usd"
) -> None:
    """Log a USD visual stage as static Rerun meshes."""

    for mesh in iter_usd_meshes(usd_path, root_path=root_path):
        kwargs: dict[str, Any] = {
            "vertex_positions": mesh.vertex_positions,
            "triangle_indices": mesh.triangle_indices,
        }
        if mesh.vertex_colors is not None:
            kwargs["vertex_colors"] = mesh.vertex_colors
        if mesh.vertex_texcoords is not None:
            kwargs["vertex_texcoords"] = mesh.vertex_texcoords
        if mesh.albedo_texture is not None:
            kwargs["albedo_texture"] = mesh.albedo_texture
        if mesh.albedo_factor is not None:
            kwargs["albedo_factor"] = mesh.albedo_factor
        recording.log(
            mesh.entity_path,
            rr.Mesh3D(**kwargs),
            static=True,
        )
