"""Fakes and fixtures for testing the Genesis -> Rerun bridge without Genesis.

The bridge is duck-typed against Genesis' scene/entity/link/sensor objects, so the
whole sensor layer can be exercised on any machine (no GPU, no `genesis-world`
install) by passing in lightweight fakes and a recording stub that captures every
`recording.log(...)` call instead of streaming to a real Rerun viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest


@dataclass
class LoggedEntry:
    """One captured `recording.log(entity_path, archetype, **kwargs)` call."""

    path: str
    archetype: Any
    kwargs: dict[str, Any]


@dataclass
class FakeRecording:
    """Stands in for the `rerun` module / a RecordingStream.

    Captures every `log` call and records the timeline state set via `set_time`,
    so tests can assert on entity paths and archetype types without a viewer.
    """

    entries: list[LoggedEntry] = field(default_factory=list)
    times: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def log(self, path: str, archetype: Any, **kwargs: Any) -> None:
        self.entries.append(LoggedEntry(path, archetype, kwargs))

    def set_time(self, timeline: str, **kwargs: Any) -> None:
        self.times.append((timeline, kwargs))

    # convenience query helpers -------------------------------------------------
    def paths(self) -> list[str]:
        return [e.path for e in self.entries]

    def archetypes(self, name: str) -> list[LoggedEntry]:
        return [e for e in self.entries if type(e.archetype).__name__ == name]

    def with_path_suffix(self, suffix: str) -> list[LoggedEntry]:
        return [e for e in self.entries if e.path.endswith(suffix)]


class FakeLink:
    """A rigid link exposing the pose accessors `transforms.log` looks for."""

    def __init__(self, name: str, pos: Any, quat_wxyz: Any, idx_local: int = 0) -> None:
        self.name = name
        self.idx_local = idx_local
        self._pos = np.asarray(pos, dtype=float)
        self._quat = np.asarray(quat_wxyz, dtype=float)

    def get_pos(self, envs_idx: int | None = None) -> np.ndarray:
        return self._pos

    def get_quat(self, envs_idx: int | None = None) -> np.ndarray:
        return self._quat


class FakeEntity:
    """A Genesis entity: a named bag of links (no morph -> no mesh logging)."""

    def __init__(self, name: str, links: list[FakeLink], idx: int = 0) -> None:
        self.name = name
        self.idx = idx
        self.links = links


class FakeScene:
    """A scene holding entities and (optionally) visualizer cameras."""

    def __init__(
        self, entities: list[FakeEntity] | None = None, cameras: list[Any] | None = None
    ) -> None:
        self.entities = entities or []
        self.cameras = cameras or []


class FakeCamera:
    """A pinhole camera that returns canned RGB/depth frames."""

    def __init__(self, rgb: np.ndarray, depth: np.ndarray, pos=(0.0, 0.0, 1.0),
                 lookat=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), fov: float = 60.0) -> None:
        self._rgb = rgb
        self._depth = depth
        self.pos = np.asarray(pos, dtype=float)
        self.lookat = np.asarray(lookat, dtype=float)
        self.up = np.asarray(up, dtype=float)
        self.fov = fov

    def render(self, rgb: bool = True, depth: bool = True):
        return (self._rgb if rgb else None, self._depth if depth else None, None, None)


@dataclass
class FakeReading:
    """A namedtuple-like sensor reading (IMU / tactile style)."""

    fields: dict[str, Any]

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple delegation
        try:
            return self.fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class FakeArraySensor:
    """A contact-force style sensor whose `read()` returns a bare 3-vector."""

    def __init__(self, vector: Any) -> None:
        self._vector = np.asarray(vector, dtype=float)

    def read(self) -> np.ndarray:
        return self._vector


class FakeFieldSensor:
    """A sensor whose `read()` returns a reading object with named fields."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def read(self) -> FakeReading:
        return FakeReading(self._fields)


class TorchLikeTensor:
    """Mimics a CUDA torch tensor: detach -> cpu -> numpy."""

    def __init__(self, array: Any) -> None:
        self._array = np.asarray(array)

    def detach(self) -> TorchLikeTensor:
        return self

    def cpu(self) -> TorchLikeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


@pytest.fixture
def recording() -> FakeRecording:
    return FakeRecording()
