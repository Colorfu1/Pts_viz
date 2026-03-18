"""Point loading adapters for the standalone browser viewer project."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np


def _resolve_custom_loader():
    loader_spec = os.getenv("PKL_POINTCLOUD_VIEWER_LOADER", "").strip()
    if loader_spec:
        module_name, sep, func_name = loader_spec.partition(":")
        if not sep:
            raise ValueError(
                "PKL_POINTCLOUD_VIEWER_LOADER must look like 'package.module:function'"
            )
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    for module_name in ("custom_point_loader", ".custom_point_loader"):
        try:
            if module_name.startswith("."):
                module = importlib.import_module(module_name, package=__package__)
            else:
                module = importlib.import_module(module_name)
        except ImportError:
            continue
        return getattr(module, "load_points", None)
    return None


def _load_points_from_path(path_like: str | Path) -> np.ndarray:
    path = Path(path_like).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Point file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            for key in ("points", "pts", "lidar", "xyz"):
                if key in payload:
                    points = payload[key]
                    break
            else:
                first_key = next(iter(payload.files), None)
                if first_key is None:
                    raise ValueError(f"No arrays found in {path}")
                points = payload[first_key]
    elif suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 6 == 0:
            points = raw.reshape(-1, 6)
        elif raw.size % 5 == 0:
            points = raw.reshape(-1, 5)
        elif raw.size % 4 == 0:
            points = raw.reshape(-1, 4)
        elif raw.size % 3 == 0:
            points = raw.reshape(-1, 3)
        else:
            raise ValueError(f"Cannot infer point shape from {path}")
    else:
        raise ValueError(
            f"Unsupported point file suffix '{suffix}' for {path}. "
            "Built-in loader supports .npy, .npz, and float32 .bin files."
        )

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected NxC point array with C>=3, got {points.shape} from {path}")
    return points


def _load_builtin_points(source: Any) -> np.ndarray:
    if isinstance(source, (str, Path)):
        return _load_points_from_path(source)

    if isinstance(source, (list, tuple)):
        arrays = [_load_builtin_points(item) for item in source]
        arrays = [item for item in arrays if item.size]
        if not arrays:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(arrays, axis=0)

    if isinstance(source, dict):
        arrays = [_load_builtin_points(item) for item in source.values()]
        arrays = [item for item in arrays if item.size]
        if not arrays:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(arrays, axis=0)

    raise TypeError(f"Unsupported point source type: {type(source)!r}")


def load_points(lidar_source: Any, at720: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load points for one frame.

    Resolution order:
    1. `PKL_POINTCLOUD_VIEWER_LOADER=package.module:function`
    2. Local `custom_point_loader.py` with `load_points(...)`
    3. Built-in file loader for path-based sources
    """

    custom_loader = _resolve_custom_loader()
    if custom_loader is not None:
        return custom_loader(lidar_source, at720=at720)

    try:
        points = _load_builtin_points(lidar_source)
    except Exception as exc:
        raise RuntimeError(
            "No usable point loader found. Add custom_point_loader.py in the project root "
            "or set PKL_POINTCLOUD_VIEWER_LOADER=package.module:function."
        ) from exc

    valid_mask = np.isfinite(points[:, :3]).all(axis=1)
    valid_mask &= (np.abs(points[:, :3]) < 1000).all(axis=1)
    return points[valid_mask], valid_mask
