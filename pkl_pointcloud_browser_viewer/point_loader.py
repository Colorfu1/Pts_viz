"""Frame/bundle loading adapters for the browser viewer project."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType | None:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        return None
    return module


def _iter_loader_modules() -> Iterator[ModuleType]:
    seen: set[str] = set()

    loader_spec = os.getenv("PKL_POINTCLOUD_VIEWER_LOADER", "").strip()
    if loader_spec:
        module_name, sep, _ = loader_spec.partition(":")
        if not sep:
            raise ValueError(
                "PKL_POINTCLOUD_VIEWER_LOADER must look like 'package.module:function'"
            )
        if module_name not in seen:
            seen.add(module_name)
            yield importlib.import_module(module_name)

    custom_loader_path = PROJECT_ROOT / "custom_point_loader.py"
    if custom_loader_path.exists():
        module_name = "project_custom_point_loader"
        if module_name not in seen:
            seen.add(module_name)
            module = _load_module_from_path(module_name, custom_loader_path)
            if module is not None:
                yield module

    for module_name in (
        "custom_point_loader",
        ".custom_point_loader",
        "pkl_pointcloud_browser_viewer.demo_point_loader",
        ".demo_point_loader",
    ):
        canonical_name = module_name.lstrip(".")
        if canonical_name in seen:
            continue
        try:
            if module_name.startswith("."):
                module = importlib.import_module(module_name, package=__package__)
            else:
                module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        seen.add(canonical_name)
        yield module


def _resolve_env_callable() -> tuple[Any, str] | tuple[None, None]:
    loader_spec = os.getenv("PKL_POINTCLOUD_VIEWER_LOADER", "").strip()
    if not loader_spec:
        return None, None
    module_name, sep, func_name = loader_spec.partition(":")
    if not sep:
        raise ValueError(
            "PKL_POINTCLOUD_VIEWER_LOADER must look like 'package.module:function'"
        )
    module = importlib.import_module(module_name)
    return getattr(module, func_name), func_name


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


def load_frame_bundle(
    source_file: str | Path,
    *,
    eval_dir: str | Path | None = None,
    at720: bool = False,
) -> dict[str, Any] | None:
    env_callable, env_name = _resolve_env_callable()
    if env_callable is not None and env_name == "load_frame_bundle":
        bundle = env_callable(source_file=source_file, eval_dir=eval_dir, at720=at720)
        if bundle is not None:
            return bundle

    for module in _iter_loader_modules():
        load_bundle = getattr(module, "load_frame_bundle", None)
        if load_bundle is None:
            continue
        bundle = load_bundle(source_file=source_file, eval_dir=eval_dir, at720=at720)
        if bundle is not None:
            return bundle
    return None


def load_points(lidar_source: Any, at720: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load points for one frame.

    Resolution order:
    1. `PKL_POINTCLOUD_VIEWER_LOADER=package.module:function`
    2. Local `custom_point_loader.py` with `load_points(...)`
    3. Built-in file loader for path-based sources
    """

    env_callable, env_name = _resolve_env_callable()
    if env_callable is not None and env_name == "load_points":
        result = env_callable(lidar_source, at720=at720)
        if result is not None:
            return result

    for module in _iter_loader_modules():
        custom_loader = getattr(module, "load_points", None)
        if custom_loader is None:
            continue
        result = custom_loader(lidar_source, at720=at720)
        if result is not None:
            return result

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
