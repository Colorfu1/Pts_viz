#!/usr/bin/env python3
"""
Browser point cloud viewer for PKL-indexed frame datasets used by
`mi_pyvista_vis_multi.py`.

The top-level PKL is treated as a lightweight frame index. Actual per-frame
pickles, lidar points, labels, detections, flow, and static labels are loaded
only when the browser requests a specific frame.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import threading
import webbrowser
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = SCRIPT_DIR / "vendor" / "browser_viewer"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "viewer.yaml"
DEFAULT_EDIT_RECTANGLES_PATH = SCRIPT_DIR / "config" / "edit_rectangles.json"


PTS_SEG_ID_AND_CLS = {
    0: "Ground",
    1: "Construction Others",
    2: "Construction Pillars",
    3: "Curb",
    4: "Car",
    5: "Bus",
    6: "Car_truck",
    7: "PD",
    8: "Cyclist",
    9: "Obstacle in lane",
    10: "Noise",
    11: "Sign",
    12: "Others",
    13: "Ignore",
    14: "Animal",
}

CLASSNAME_TO_COLOR = np.array(
    [
        (255, 255, 255),  # Ground
        (255, 99, 71),  # Construction Others
        (255, 20, 147),  # Construction Pillars
        (0, 191, 255),  # Curb
        (84, 255, 159),  # Car
        (65, 105, 225),  # Bus
        (47, 79, 79),  # Car_truck
        (255, 255, 0),  # PD
        (255, 140, 0),  # Cyclist
        (255, 0, 255),  # Obstacle in lane
        (255, 218, 185),  # Noise
        (0, 255, 255),  # Sign
        (188, 143, 143),  # Others
        (128, 128, 128),  # Ignore
        (186, 85, 211),  # Animal
    ],
    dtype=np.uint8,
)

BBOX_CLASS_TO_SEG_CLASS = {
    0: 4,
    1: 6,
    2: 5,
    3: 7,
    4: 8,
    5: 2,
    6: 14,
}

BBOX_CLASS_NAMES = ["car", "truck", "bus", "pedestrian", "cyclist", "barrier", "animal"]


def save_edit_rectangles(path: str | Path, rectangles: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, list[list[float]]] = {}
    if not isinstance(rectangles, dict):
        raise ValueError("rectangles must be a dict")
    for frame_name, points in rectangles.items():
        if not isinstance(frame_name, str) or not frame_name:
            raise ValueError("rectangle key must be a non-empty string")
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError(f"{frame_name}: rectangle must have exactly 4 points")
        normalized_points: list[list[float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"{frame_name}: each point must be [x, y]")
            x = float(point[0])
            y = float(point[1])
            normalized_points.append([x, y])
        normalized[frame_name] = normalized_points

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"path": str(output_path), "count": len(normalized)}


def _iter_frame_adrns(data_info: dict[str, Any]) -> list[str]:
    raw_adrns = data_info.get("adrns")
    if raw_adrns is None:
        return []
    if isinstance(raw_adrns, dict):
        values: list[Any] = list(raw_adrns.values())
    elif isinstance(raw_adrns, (list, tuple)):
        values = list(raw_adrns)
    else:
        values = [raw_adrns]
    return [str(value) for value in values if value]


def _is_image_adrn(adrn: str) -> bool:
    value = str(adrn)
    if "image" in value:
        return True
    if ":cam." not in value:
        return False
    tail = value.rsplit(":", 1)[-1].strip()
    return bool(tail) and tail not in {"-1", "meta"}


def _load_matching_annotation_adrns(data_info: dict[str, Any]) -> list[str]:
    anno_file = data_info.get("anno_file")
    frame_adrn = str(data_info.get("adrn") or "").strip()
    if not anno_file or not frame_adrn:
        return []
    anno_path = Path(str(anno_file)).expanduser()
    if not anno_path.exists():
        return []
    with open(anno_path, "rb") as handle:
        anno_payload = pickle.load(handle)

    def find_matching_adrns(entries: Any) -> list[str]:
        if not isinstance(entries, list):
            return []
        for item in entries:
            if not isinstance(item, dict):
                continue
            if str(item.get("adrn") or "").strip() != frame_adrn:
                continue
            return _iter_frame_adrns(item)
        return []

    matched_adrns = find_matching_adrns(anno_payload.get("annotation"))
    if matched_adrns:
        return matched_adrns
    matched_adrns = find_matching_adrns(anno_payload.get("od_labeled_frames"))
    if matched_adrns:
        return matched_adrns
    return []


def _extract_camera_key_from_adrn(adrn: str) -> str | None:
    parts = str(adrn).split(":")
    for part in parts:
        if not part.startswith("cam."):
            continue
        camera_part = part.removeprefix("cam.")
        camera_tokens = camera_part.split(".")
        if len(camera_tokens) >= 2 and camera_tokens[-1].isdigit():
            camera_tokens = camera_tokens[:-1]
        camera_key = ".".join(camera_tokens).replace(".", "_").strip("_")
        return camera_key or None
    return None


def extract_frame_image_sources(data_info: dict[str, Any]) -> list[dict[str, str]]:
    candidate_adrns = _iter_frame_adrns(data_info)
    if not any(_is_image_adrn(adrn) for adrn in candidate_adrns):
        candidate_adrns = _load_matching_annotation_adrns(data_info)
    image_sources: list[dict[str, str]] = []
    seen_camera_keys: set[str] = set()
    for adrn in candidate_adrns:
        if not _is_image_adrn(adrn):
            continue
        camera_key = _extract_camera_key_from_adrn(adrn)
        if not camera_key or camera_key in seen_camera_keys:
            continue
        image_sources.append(
            {
                "cameraKey": camera_key,
                "cameraLabel": camera_key,
                "adrn": adrn,
            }
        )
        seen_camera_keys.add(camera_key)
    return image_sources


def resolve_frame_image_source(data_info: dict[str, Any], camera_key: str) -> dict[str, str] | None:
    normalized_camera_key = str(camera_key or "").strip()
    if not normalized_camera_key:
        return None
    for source in extract_frame_image_sources(data_info):
        if source["cameraKey"] == normalized_camera_key:
            return source
    return None


def normalize_image_array_for_browser(image_array: np.ndarray) -> np.ndarray:
    array = np.asarray(image_array)
    if array.ndim == 3 and array.shape[-1] == 3:
        # bytes_to_image_array follows OpenCV-style BGR channel order.
        return np.ascontiguousarray(array[..., ::-1])
    if array.ndim == 3 and array.shape[-1] == 4:
        # Preserve alpha while converting BGRA to RGBA.
        return np.ascontiguousarray(array[..., [2, 1, 0, 3]])
    return np.ascontiguousarray(array)


def load_points_for_frame(lidar_adrn: Any, at720: bool) -> tuple[np.ndarray | None, np.ndarray | None]:
    from .point_loader import load_points

    return load_points(lidar_adrn, at720=at720)


def load_frame_bundle_for_source(
    source_file: str | Path,
    *,
    eval_dir: str | Path | None,
    at720: bool,
) -> dict[str, Any] | None:
    from .point_loader import load_frame_bundle

    return load_frame_bundle(source_file, eval_dir=eval_dir, at720=at720)


def _load_image_bytes_for_adrn(adrn: str) -> bytes | None:
    try:
        from .point_loader import _iter_loader_modules
    except Exception:
        return None

    for module in _iter_loader_modules():
        reader = getattr(module, "simple_read_frame", None)
        if reader is None:
            continue
        try:
            image_bytes = reader(adrn)
        except TypeError:
            image_bytes = None
        except Exception:
            image_bytes = None
        if image_bytes is not None:
            return image_bytes
    return None
BBOX_NAME_TO_CLASS = {name: idx for idx, name in enumerate(BBOX_CLASS_NAMES)}

CATEMAP = {
    "car": "car",
    "sport_utility_vehicle": "car",
    "pickup": "car",
    "dummy_car": "car",
    "van": "car",
    "bus": "bus",
    "truck_with_container": "truck",
    "truck": "truck",
    "trailer": "truck",
    "other_vehicle": "truck",
    "trailer_body": "truck",
    "trailer_head": "truck",
    "trailer_with_container": "truck",
    "tanker": "truck",
    "oil_trailer": "truck",
    "truck_head": "truck",
    "truck_body": "truck",
    "cyclist": "cyclist",
    "dummy_cyclist": "cyclist",
    "bicycle_without_person": "cyclist",
    "parked_cycle": "cyclist",
    "handcart": "cyclist",
    "handcart_stoped": "cyclist",
    "stroller": "cyclist",
    "stroller_stoped": "cyclist",
    "motorcyclist": "cyclist",
    "tricyclist": "cyclist",
    "motorcycle_without_person": "cyclist",
    "tricycle_without_person": "cyclist",
    "pedestrian": "pedestrian",
    "dummy": "pedestrian",
    "people": "pedestrian",
    "animal": "animal",
    "barrier": "barrier",
    "pillar": "barrier",
    "traffic_cone": "barrier",
    "crash_pile": "barrier",
    "water_filled_barrier": "barrier",
    "road_safety_barrel": "barrier",
    "other_obstacle": "barrier",
    "warning_triangle": "barrier",
    "ignore": "ignore",
    "unknown": "ignore",
    "car_door": "ignore",
}

PT_SEG_MAP_DICT = np.array(
    [
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 9, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        0, 12, 12, 12, 12, 12, 12, 12, 12, 3,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        10, 10, 10, 10, 10, 12, 12, 12, 12, 12,
        12, 12, 12, 2, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        4, 4, 4, 5, 6, 6, 8, 8, 8, 8,
        7, 1, 7, 9, 6, 6, 6, 8, 6, 1,
        12, 12, 2, 2, 2, 2, 12, 12, 12, 12,
        12, 12, 12, 8, 8, 6, 6, 8, 8, 12,
        8, 4, 1, 1, 6, 6, 6, 8, 8, 8,
        8, 6, 4, 6, 9, 9, 12, 9, 11, 9,
        12, 12, 12, 9, 9, 13, 13,
    ],
    dtype=np.int32,
)

PC_RANGE = (-24.0, -80.0, -10.0, 24.0, 200.0, 10.0)


def _srgb_channel_to_linear_u8(values: np.ndarray) -> np.ndarray:
    normalized = values.astype(np.float32) / 255.0
    linear = np.where(
        normalized <= 0.04045,
        normalized / 12.92,
        ((normalized + 0.055) / 1.055) ** 2.4,
    )
    return np.clip(np.round(linear * 255.0), 0, 255).astype(np.uint8)


RENDER_CLASSNAME_TO_COLOR = _srgb_channel_to_linear_u8(CLASSNAME_TO_COLOR)
STATIC_LABEL_COLORS = np.ascontiguousarray(
    _srgb_channel_to_linear_u8(
        np.array(
            [
                (255, 255, 255),  # static
                (255, 99, 71),    # dynamic
            ],
            dtype=np.uint8,
        )
    ),
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve PKL-indexed point cloud results in a browser."
    )
    parser.add_argument(
        "pkl_file",
        nargs="?",
        default=None,
        help="Top-level PKL that indexes per-frame PKLs",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config file. CLI args override config values.",
    )
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Optional eval directory containing per-frame prediction PKLs",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="HTTP bind host. Use 0.0.0.0 for remote access.",
    )
    parser.add_argument("--port", type=int, default=None, help="HTTP port")
    parser.add_argument(
        "--fps", type=float, default=None, help="Default browser playback FPS"
    )
    parser.add_argument(
        "--point-size", type=float, default=None, help="Default rendered point size"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Keep every Nth point to reduce browser load",
    )
    parser.add_argument(
        "--label-source",
        choices=["gt", "pred"],
        default=None,
        help="Segmentation label source when prediction PKLs are available",
    )
    parser.add_argument(
        "--at720",
        dest="at720",
        action="store_true",
        default=None,
        help="Forward the at720 flag to the lidar point loader",
    )
    parser.add_argument(
        "--no-at720",
        dest="at720",
        action="store_false",
        help="Disable the at720 flag even if the config enables it",
    )
    parser.add_argument(
        "--open-browser",
        dest="open_browser",
        action="store_true",
        default=None,
        help="Open the page in the default browser after startup",
    )
    parser.add_argument(
        "--no-open-browser",
        dest="open_browser",
        action="store_false",
        help="Disable browser auto-open even if the config enables it",
    )

    args = parser.parse_args()
    config_path = Path(args.config).expanduser()
    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            parser.error(f"Config file must contain a mapping: {config_path}")

    defaults = {
        "eval_dir": "",
        "host": "127.0.0.1",
        "port": 8766,
        "fps": 5.0,
        "point_size": 2.0,
        "sample_rate": 1,
        "label_source": "gt",
        "at720": False,
        "open_browser": False,
    }

    def pick(name: str):
        value = getattr(args, name)
        if value is not None:
            return value
        return config.get(name, defaults[name])

    args.pkl_file = args.pkl_file or config.get("pkl_file")
    if not args.pkl_file:
        parser.error(
            f"Missing pkl_file. Set it in {config_path} or pass it on the command line."
        )

    args.eval_dir = pick("eval_dir")
    args.host = pick("host")
    args.port = int(pick("port"))
    args.fps = float(pick("fps"))
    args.point_size = float(pick("point_size"))
    args.sample_rate = int(pick("sample_rate"))
    args.label_source = pick("label_source")
    if args.label_source not in {"gt", "pred"}:
        parser.error(f"label_source must be 'gt' or 'pred', got: {args.label_source}")
    args.at720 = bool(pick("at720"))
    args.open_browser = bool(pick("open_browser"))
    args.config = str(config_path)
    return args


class FrameStore:
    def __init__(
        self,
        pkl_file: str,
        eval_dir: str,
        sample_rate: int,
        label_source: str,
        at720: bool,
    ):
        self.pkl_file = Path(pkl_file).expanduser().resolve()
        if not self.pkl_file.exists():
            raise FileNotFoundError(f"PKL file does not exist: {self.pkl_file}")
        self.eval_dir = None
        if eval_dir:
            candidate = Path(eval_dir).expanduser().resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"Eval directory does not exist: {candidate}")
            if not candidate.is_dir():
                raise NotADirectoryError(f"Eval path is not a directory: {candidate}")
            self.eval_dir = candidate
        self.data_dir = self.pkl_file.parent
        self.sample_rate = max(1, sample_rate)
        self.label_source = label_source
        self.at720 = at720
        self.frame_paths = self._load_frame_index()
        if not self.frame_paths:
            raise FileNotFoundError(f"No frame entries found in {self.pkl_file}")
        (
            self.has_pred_modality,
            self.has_flow_modality,
            self.has_static_modality,
            self.has_point_loss_modality,
        ) = self._detect_optional_modalities()

    def __len__(self) -> int:
        return len(self.frame_paths)

    @staticmethod
    def _normalize_label(label: np.ndarray) -> np.ndarray:
        label = np.asarray(label)
        if label.ndim == 2:
            label = np.squeeze(label, axis=-1)
        label = label.astype(np.float32, copy=False)
        max_class_id = len(CLASSNAME_TO_COLOR) - 1
        if label.size == 0:
            return label.astype(np.int32)
        if np.max(label) > max_class_id:
            min_value = float(np.min(label))
            max_value = float(np.max(label))
            if max_value > min_value:
                label = (label - min_value) / (max_value - min_value) * max_class_id
            else:
                label = np.zeros_like(label)
        return np.clip(label.astype(np.int32), 0, max_class_id)

    @staticmethod
    def _load_label(seg_path: Path) -> np.ndarray | None:
        if not seg_path.exists():
            return None
        label = np.load(seg_path, allow_pickle=False)
        return FrameStore._normalize_label(label)

    @staticmethod
    def _normalize_name(name: Any) -> str:
        mapped = CATEMAP.get(str(name), str(name))
        return mapped

    def _load_frame_index(self) -> list[Path]:
        with open(self.pkl_file, "rb") as handle:
            data = pickle.load(handle)
        frame_refs = []
        for key in ("l2", "l3", "infos"):
            values = data.get(key, [])
            if values:
                frame_refs.extend(values)
        frame_paths: list[Path] = []
        for entry in frame_refs:
            path = Path(str(entry)).expanduser()
            if not path.is_absolute():
                path = self.data_dir / path
            frame_paths.append(path)
        return frame_paths

    def _frame_info_path(self, index: int) -> Path:
        return self.frame_paths[index]

    def _load_frame_info(self, index: int) -> dict[str, Any]:
        frame_info_path = self._frame_info_path(index)
        with open(frame_info_path, "rb") as handle:
            return pickle.load(handle)

    def _prediction_path(self, data_info: dict[str, Any]) -> Path | None:
        if self.eval_dir is None:
            return None
        adrn_value = data_info.get("adrn", "")
        if isinstance(adrn_value, dict):
            adrn_value = adrn_value.get("Major", next(iter(adrn_value.values()), ""))
        if isinstance(adrn_value, list):
            chosen = None
            for item in adrn_value:
                if "lidar.mid_center_top_wide" in item:
                    chosen = item
                    break
            adrn_value = chosen or (adrn_value[0] if adrn_value else "")
        if not adrn_value:
            return None
        return self.eval_dir / (str(adrn_value).replace(":", "_") + ".pkl")

    @staticmethod
    def _point_range_mask(points: np.ndarray) -> np.ndarray:
        x_min, y_min, z_min, x_max, y_max, z_max = PC_RANGE
        return (
            (points[:, 0] > x_min)
            & (points[:, 0] < x_max)
            & (points[:, 1] > y_min)
            & (points[:, 1] < y_max)
            & (points[:, 2] > z_min)
            & (points[:, 2] < z_max)
        )

    @staticmethod
    def _map_raw_gt_labels(raw_labels: np.ndarray) -> np.ndarray:
        raw_labels = np.asarray(raw_labels).reshape(-1).astype(np.int32, copy=False)
        mapped = np.full(raw_labels.shape, 12, dtype=np.int32)
        valid = (raw_labels >= 0) & (raw_labels < len(PT_SEG_MAP_DICT))
        mapped[valid] = PT_SEG_MAP_DICT[raw_labels[valid]]
        return mapped

    @staticmethod
    def _ensure_2d_array(values: np.ndarray, name: str, min_cols: int) -> np.ndarray:
        array = np.asarray(values)
        if array.size == 0:
            return np.empty((0, min_cols), dtype=np.float32)
        if array.ndim == 1:
            array = np.expand_dims(array, axis=0)
        if array.ndim != 2 or array.shape[1] < min_cols:
            raise ValueError(f"Unexpected {name} shape: {array.shape}")
        return array

    @staticmethod
    def _apply_sample_rate(
        positions: np.ndarray,
        gt_labels: np.ndarray,
        pred_labels: np.ndarray | None,
        flow: np.ndarray | None,
        static_labels: np.ndarray | None,
        point_loss: np.ndarray | None,
        ignore_mask: np.ndarray | None,
        sample_rate: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        if sample_rate <= 1:
            return positions, gt_labels, pred_labels, flow, static_labels, point_loss, ignore_mask
        positions = positions[::sample_rate]
        gt_labels = gt_labels[::sample_rate]
        if pred_labels is not None:
            pred_labels = pred_labels[::sample_rate]
        if flow is not None:
            flow = flow[::sample_rate]
        if static_labels is not None:
            static_labels = static_labels[::sample_rate]
        if point_loss is not None:
            point_loss = point_loss[::sample_rate]
        if ignore_mask is not None:
            ignore_mask = ignore_mask[::sample_rate]
        return positions, gt_labels, pred_labels, flow, static_labels, point_loss, ignore_mask

    @staticmethod
    def _align_pred_array(
        values: np.ndarray | None,
        count: int | None,
        pre_range_len: int,
        range_mask: np.ndarray,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values)
        if count is not None:
            array = array[:count]
        if len(array) == int(np.sum(range_mask)):
            return array
        if len(array) == pre_range_len:
            return array[range_mask]
        raise ValueError(f"{name} length mismatch: {len(array)} vs expected {int(np.sum(range_mask))}")

    @staticmethod
    def _compute_scalar_range(
        values: np.ndarray | None, valid_mask: np.ndarray | None = None
    ) -> tuple[float, float] | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if valid_mask is not None:
            mask = np.asarray(valid_mask).reshape(-1).astype(bool, copy=False)
            if mask.shape[0] == array.shape[0]:
                array = array[mask]
        if array.size == 0:
            return None
        return float(np.min(array)), float(np.max(array))

    @staticmethod
    def _get_gt_flow_mask(data_info: dict[str, Any]) -> np.ndarray | None:
        raw_mask = data_info.get("flow_maks")
        if raw_mask is None:
            raw_mask = data_info.get("flow_mask")
        if raw_mask is None:
            return None
        return np.asarray(raw_mask)

    def _load_gt_flow_values(
        self,
        data_info: dict[str, Any],
        point_selector: np.ndarray,
        range_mask: np.ndarray,
    ) -> np.ndarray | None:
        raw_flow = data_info.get("flow_gt")
        if raw_flow is None:
            return None
        flow_values = self._ensure_2d_array(raw_flow, "flow_gt", 3)[point_selector][range_mask]
        raw_mask = self._get_gt_flow_mask(data_info)
        if raw_mask is None:
            return flow_values
        flow_mask = np.asarray(raw_mask)[point_selector][range_mask].reshape(-1).astype(bool, copy=False)
        if flow_mask.shape[0] != flow_values.shape[0]:
            raise ValueError(
                f"flow_mask length mismatch: {flow_mask.shape[0]} vs expected {flow_values.shape[0]}"
            )
        flow_values = np.array(flow_values, copy=True)
        flow_values[~flow_mask] = 0
        return flow_values

    def _load_gt_static_labels(
        self,
        data_info: dict[str, Any],
        point_selector: np.ndarray,
        range_mask: np.ndarray,
    ) -> np.ndarray | None:
        flow_values = self._load_gt_flow_values(data_info, point_selector, range_mask)
        if flow_values is None:
            return None
        static_labels = np.all(flow_values[:, :3] == 0.0, axis=1).astype(np.uint8, copy=False)
        return np.ascontiguousarray(static_labels.astype(np.uint8, copy=False))

    @staticmethod
    def _normalize_pose_matrix(pose: Any, name: str) -> np.ndarray:
        pose_array = np.asarray(pose, dtype=np.float32)
        if pose_array.shape != (4, 4):
            raise ValueError(f"Unexpected {name} shape: {pose_array.shape}")
        return pose_array

    @staticmethod
    def _extract_xyz(position: Any) -> list[float]:
        if isinstance(position, dict):
            return [float(position["x"]), float(position["y"]), float(position["z"])]
        pos = np.asarray(position, dtype=np.float32).reshape(-1)
        if pos.shape[0] < 3:
            raise ValueError(f"Unexpected position shape: {pos.shape}")
        return [float(pos[0]), float(pos[1]), float(pos[2])]

    @staticmethod
    def _transform_boxes_to_current_frame(
        boxes: np.ndarray,
        ego2global: np.ndarray,
        next_pose: np.ndarray,
    ) -> np.ndarray:
        final_trans = np.linalg.inv(ego2global) @ next_pose
        xyz = boxes[:, :3]
        xyz = np.hstack((xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)))
        xyz = (final_trans @ xyz.T).T
        transformed = np.array(boxes, copy=True)
        transformed[:, :3] = xyz[:, :3]
        return transformed

    @staticmethod
    def _transform_points_to_current_frame(
        points_xyz: np.ndarray,
        ego2global: np.ndarray,
        sweep_pose: np.ndarray,
    ) -> np.ndarray:
        final_trans = np.linalg.inv(ego2global) @ sweep_pose
        xyz = np.hstack((points_xyz[:, :3], np.ones((points_xyz.shape[0], 1), dtype=np.float32)))
        xyz = (final_trans @ xyz.T).T
        return np.ascontiguousarray(xyz[:, :3], dtype=np.float32)

    def _build_named_detections(
        self,
        boxes: np.ndarray | None,
        names: list[Any] | np.ndarray | None,
        *,
        is_next_frame: bool = False,
    ) -> list[dict]:
        if boxes is None or names is None:
            return []
        boxes = self._ensure_2d_array(boxes, "gt_boxes", 7)
        names = list(names)
        detections = []
        for row, raw_name in zip(boxes, names):
            name = self._normalize_name(raw_name)
            bbox_class = BBOX_NAME_TO_CLASS.get(name)
            if bbox_class is None:
                continue
            seg_class = BBOX_CLASS_TO_SEG_CLASS[bbox_class]
            detections.append(
                {
                    "center": [float(row[0]), float(row[1]), float(row[2])],
                    "size": [float(row[3]), float(row[4]), float(row[5])],
                    "yaw": float(row[6]),
                    "score": 1.0,
                    "bboxClass": bbox_class,
                    "segClass": seg_class,
                    "name": name,
                    "color": RENDER_CLASSNAME_TO_COLOR[seg_class].tolist(),
                    "isNextFrame": is_next_frame,
                }
            )
        return detections

    def _build_gt_detections(self, data_info: dict[str, Any]) -> list[dict]:
        return self._build_named_detections(
            data_info.get("gt_boxes"),
            data_info.get("gt_names"),
            is_next_frame=False,
        )

    def _build_next_gt_detections(self, data_info: dict[str, Any]) -> list[dict]:
        next_labeled_frame = data_info.get("next_labeled_frame")
        if not isinstance(next_labeled_frame, dict):
            return []
        next_gt = next_labeled_frame.get("gt")
        if not next_gt:
            return []
        ego2global = data_info.get("ego2global")
        next_pose = next_labeled_frame.get("pose")
        if ego2global is None or next_pose is None:
            return []
        ego2global = self._normalize_pose_matrix(ego2global, "ego2global")
        next_pose = self._normalize_pose_matrix(next_pose, "next_labeled_frame.pose")

        next_boxes = []
        next_names = []
        for item in next_gt:
            if not isinstance(item, dict):
                continue
            try:
                xyz = self._extract_xyz(item.get("position"))
                width = float(item.get("width", 0.0))
                length = float(item.get("length", 0.0))
                height = float(item.get("height", 0.0))
                yaw = float(item.get("theta", item.get("orientation", 0.0)))
            except (KeyError, TypeError, ValueError):
                continue
            next_boxes.append(xyz + [width, length, height, yaw])
            next_names.append(item.get("name"))

        if not next_boxes:
            return []

        next_boxes_array = np.asarray(next_boxes, dtype=np.float32)
        next_boxes_array = self._transform_boxes_to_current_frame(next_boxes_array, ego2global, next_pose)
        next_boxes_array[:, [3, 4]] = next_boxes_array[:, [4, 3]]
        return self._build_named_detections(
            next_boxes_array,
            next_names,
            is_next_frame=True,
        )

    def _build_pred_detections(self, pred_data: dict[str, Any] | None) -> list[dict]:
        if not pred_data:
            return []
        voxel_results = pred_data.get("voxel_results")
        if not isinstance(voxel_results, dict):
            return []
        if "boxes_3d" not in voxel_results or "labels_3d" not in voxel_results:
            return []
        boxes = self._ensure_2d_array(voxel_results["boxes_3d"], "boxes_3d", 7).astype(np.float32)
        labels = np.asarray(voxel_results["labels_3d"]).reshape(-1).astype(np.int32, copy=False)
        scores = np.asarray(voxel_results.get("scores_3d", np.ones(len(boxes)))).reshape(-1).astype(np.float32, copy=False)
        valid = scores > 0.1
        boxes = boxes[valid]
        labels = labels[valid]
        scores = scores[valid]
        detections = []
        for row, bbox_class, score in zip(boxes, labels, scores):
            if bbox_class < 0 or bbox_class >= len(BBOX_CLASS_NAMES):
                continue
            seg_class = BBOX_CLASS_TO_SEG_CLASS[int(bbox_class)]
            detections.append(
                {
                    "center": [float(row[0]), float(row[1]), float(row[2] + row[5] / 2.0)],
                    "size": [float(row[3]), float(row[4]), float(row[5])],
                    "yaw": float(row[6]),
                    "score": float(score),
                    "bboxClass": int(bbox_class),
                    "segClass": seg_class,
                    "name": BBOX_CLASS_NAMES[int(bbox_class)],
                    "color": RENDER_CLASSNAME_TO_COLOR[seg_class].tolist(),
                }
            )
        return detections

    def _normalize_external_detections(self, detections: Any) -> list[dict]:
        normalized: list[dict] = []
        if not detections:
            return normalized
        for item in detections:
            if not isinstance(item, dict):
                continue
            detection = dict(item)
            seg_class = int(detection.get("segClass", 12))
            detection["segClass"] = seg_class
            detection["bboxClass"] = int(detection.get("bboxClass", -1))
            detection["score"] = float(detection.get("score", 1.0))
            detection["yaw"] = float(detection.get("yaw", 0.0))
            detection["center"] = [float(x) for x in detection.get("center", [0.0, 0.0, 0.0])]
            detection["size"] = [float(x) for x in detection.get("size", [1.0, 1.0, 1.0])]
            detection["name"] = str(detection.get("name", "unknown"))
            detection["isNextFrame"] = bool(detection.get("isNextFrame", False))
            if "color" not in detection:
                detection["color"] = RENDER_CLASSNAME_TO_COLOR[seg_class].tolist()
            else:
                detection["color"] = [int(x) for x in detection["color"]]
            normalized.append(detection)
        return normalized

    def _default_external_log_info(
        self,
        frame_info_path: Path,
        *,
        positions: np.ndarray,
        pred_labels: np.ndarray | None,
        flow_values: np.ndarray | None,
        static_labels: np.ndarray | None,
        point_loss: np.ndarray | None,
        gt_detections: list[dict],
        pred_detections: list[dict],
    ) -> dict[str, Any]:
        point_loss_range = self._compute_scalar_range(point_loss)
        return {
            "entries": [
                {"key": "mode_default", "value": self.label_source},
                {"key": "frame_file_name", "value": frame_info_path.name},
                {"key": "frame_file_path", "value": str(frame_info_path)},
                {"key": "original_points", "value": len(positions)},
                {"key": "voxel_selected_points", "value": len(positions)},
                {"key": "in_range_points", "value": len(positions)},
                {"key": "ignored_points_after_downsample", "value": 0},
                {"key": "visible_points", "value": len(positions)},
                {"key": "gt_det_count", "value": len(gt_detections)},
                {"key": "eval_det_count", "value": len(pred_detections)},
                {"key": "has_pred_labels", "value": pred_labels is not None},
                {"key": "has_flow", "value": flow_values is not None},
                {"key": "has_static_labels", "value": static_labels is not None},
                {"key": "has_point_loss", "value": point_loss_range is not None},
                {"key": "point_loss_range", "value": point_loss_range},
            ]
        }

    def _normalize_external_bundle(
        self,
        frame_info_path: Path,
        bundle: dict[str, Any],
    ) -> dict[str, Any]:
        points = bundle.get("positions", bundle.get("points"))
        if points is None:
            raise ValueError(f"External bundle missing points/positions for {frame_info_path.name}")
        points = self._ensure_2d_array(points, "positions", 3).astype(np.float32, copy=False)
        positions = np.ascontiguousarray(points[:, :3], dtype=np.float32)

        gt_labels = bundle.get("gt_labels")
        if gt_labels is None:
            gt_labels = np.zeros((len(positions),), dtype=np.int32)
        gt_labels = self._normalize_label(gt_labels)
        if len(gt_labels) != len(positions):
            raise ValueError(
                f"gt_labels length mismatch for {frame_info_path.name}: {len(gt_labels)} vs {len(positions)}"
            )

        pred_labels = bundle.get("pred_labels")
        if pred_labels is not None:
            pred_labels = self._normalize_label(pred_labels)
            if len(pred_labels) != len(positions):
                raise ValueError(
                    f"pred_labels length mismatch for {frame_info_path.name}: {len(pred_labels)} vs {len(positions)}"
                )

        flow_values = bundle.get("flow")
        if flow_values is not None:
            flow_values = self._ensure_2d_array(flow_values, "flow", 3).astype(np.float32, copy=False)
            if len(flow_values) != len(positions):
                raise ValueError(
                    f"flow length mismatch for {frame_info_path.name}: {len(flow_values)} vs {len(positions)}"
                )

        static_labels = bundle.get("static_labels")
        if static_labels is not None:
            static_labels = np.asarray(static_labels).reshape(-1).astype(np.uint8, copy=False)
            if len(static_labels) != len(positions):
                raise ValueError(
                    f"static_labels length mismatch for {frame_info_path.name}: "
                    f"{len(static_labels)} vs {len(positions)}"
                )

        point_loss = bundle.get("point_loss")
        if point_loss is not None:
            point_loss = np.asarray(point_loss, dtype=np.float32).reshape(-1)
            if len(point_loss) != len(positions):
                raise ValueError(
                    f"point_loss length mismatch for {frame_info_path.name}: {len(point_loss)} vs {len(positions)}"
                )

        ignore_mask = bundle.get("ignore_mask")
        if ignore_mask is None:
            ignore_mask = np.zeros((len(positions),), dtype=np.uint8)
        else:
            ignore_mask = np.asarray(ignore_mask).reshape(-1).astype(np.uint8, copy=False)
            if len(ignore_mask) != len(positions):
                raise ValueError(
                    f"ignore_mask length mismatch for {frame_info_path.name}: {len(ignore_mask)} vs {len(positions)}"
                )

        (
            positions,
            gt_labels,
            pred_labels,
            flow_values,
            static_labels,
            point_loss,
            ignore_mask,
        ) = self._apply_sample_rate(
            positions,
            gt_labels.astype(np.int32, copy=False),
            None if pred_labels is None else pred_labels.astype(np.int32, copy=False),
            None if flow_values is None else flow_values[:, :3],
            static_labels,
            point_loss,
            ignore_mask,
            self.sample_rate,
        )

        gt_detections = self._normalize_external_detections(bundle.get("gt_detections"))
        pred_detections = self._normalize_external_detections(bundle.get("pred_detections"))
        gt_next_detections = self._normalize_external_detections(bundle.get("gt_next_detections"))
        point_loss_range = self._compute_scalar_range(
            point_loss,
            None if ignore_mask is None else (ignore_mask == 0),
        )
        log_info = bundle.get("log_info")
        if not isinstance(log_info, dict) or "entries" not in log_info:
            log_info = self._default_external_log_info(
                frame_info_path,
                positions=positions,
                pred_labels=pred_labels,
                flow_values=flow_values,
                static_labels=static_labels,
                point_loss=point_loss,
                gt_detections=gt_detections + gt_next_detections,
                pred_detections=pred_detections,
            )

        return {
            "name": str(bundle.get("name", frame_info_path.name)),
            "positions": positions,
            "gt_labels": np.ascontiguousarray(gt_labels.astype(np.int32, copy=False)),
            "pred_labels": None
            if pred_labels is None
            else np.ascontiguousarray(pred_labels.astype(np.int32, copy=False)),
            "gt_detections": gt_detections,
            "gt_next_detections": gt_next_detections,
            "pred_detections": pred_detections,
            "pred_flow": None
            if flow_values is None
            else np.ascontiguousarray(flow_values[:, :3], dtype=np.float32),
            "gt_flow": None,
            "pred_static_labels": None
            if static_labels is None
            else np.ascontiguousarray(np.clip(static_labels.astype(np.int32), 0, 1).astype(np.uint8)),
            "gt_static_labels": None,
            "point_loss": None
            if point_loss is None
            else np.ascontiguousarray(point_loss.astype(np.float32, copy=False)),
            "ignore_mask": np.ascontiguousarray(ignore_mask.astype(np.uint8, copy=False)),
            "point_loss_range": point_loss_range,
            "log_info": log_info,
        }

    @staticmethod
    def _to_log_string(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple, np.ndarray)):
            return str(list(value))
        return str(value)

    def _build_frame_log_info(
        self,
        frame_info_path: Path,
        data_info: dict[str, Any],
        pred_path: Path | None,
        raw_gt_visible: np.ndarray | None,
        mapped_gt_visible: np.ndarray,
        visible_point_count: int,
        original_point_count: int,
        voxel_selected_point_count: int,
        in_range_point_count: int,
        ignored_point_count_after_downsample: int,
        gt_det_count: int,
        pred_det_count: int,
        flow_values: np.ndarray | None,
        static_labels: np.ndarray | None,
        point_loss_range: tuple[float, float] | None,
    ) -> dict[str, Any]:
        ann_file = data_info.get("anno_file") or data_info.get("clip_name")
        raw_unique = [] if raw_gt_visible is None else np.unique(raw_gt_visible.astype(np.int32)).tolist()
        mapped_unique = np.unique(mapped_gt_visible.astype(np.int32)).tolist()
        entries = [
            {"key": "mode_default", "value": self.label_source},
            {"key": "frame_file_name", "value": frame_info_path.name},
            {"key": "frame_file_path", "value": str(frame_info_path)},
            {"key": "ann_file", "value": ann_file},
            {"key": "adrn", "value": data_info.get("adrn")},
            {"key": "eval_file", "value": None if pred_path is None else str(pred_path)},
            {"key": "original_points", "value": original_point_count},
            {"key": "voxel_selected_points", "value": voxel_selected_point_count},
            {"key": "in_range_points", "value": in_range_point_count},
            {"key": "ignored_points_after_downsample", "value": ignored_point_count_after_downsample},
            {"key": "visible_points", "value": visible_point_count},
            {"key": "unique_gt_raw", "value": raw_unique},
            {"key": "unique_gt_mapped", "value": mapped_unique},
            {"key": "gt_det_count", "value": gt_det_count},
            {"key": "eval_det_count", "value": pred_det_count},
            {"key": "has_flow", "value": flow_values is not None},
            {"key": "has_static_labels", "value": static_labels is not None},
            {"key": "has_point_loss", "value": point_loss_range is not None},
            {"key": "point_loss_range", "value": point_loss_range},
        ]
        return {
            "entries": [
                {
                    "key": item["key"],
                    "value": self._to_log_string(item["value"]),
                }
                for item in entries
            ]
        }

    def _detect_optional_modalities(self, max_checks: int = 16) -> tuple[bool, bool, bool, bool]:
        has_pred = False
        has_flow = False
        has_static = False
        has_point_loss = False
        for frame_path in self.frame_paths[:max_checks]:
            try:
                bundle = load_frame_bundle_for_source(
                    frame_path,
                    eval_dir=self.eval_dir,
                    at720=self.at720,
                )
                if bundle is not None:
                    has_pred = has_pred or (
                        bundle.get("pred_labels") is not None or bool(bundle.get("pred_detections"))
                    )
                    has_flow = has_flow or (bundle.get("flow") is not None)
                    has_static = has_static or (bundle.get("static_labels") is not None)
                    has_point_loss = has_point_loss or (bundle.get("point_loss") is not None)
                    if has_pred and has_flow and has_static and has_point_loss:
                        break
                    continue
                with open(frame_path, "rb") as handle:
                    data_info = pickle.load(handle)
                has_flow = has_flow or (data_info.get("flow_gt") is not None)
                has_static = has_static or (data_info.get("flow_gt") is not None)
                if self.eval_dir is not None:
                    pred_path = self._prediction_path(data_info)
                    if pred_path is not None and pred_path.exists():
                        with open(pred_path, "rb") as handle:
                            pred_data = pickle.load(handle)
                        has_pred = has_pred or ("pts_results" in pred_data or "voxel_results" in pred_data)
                        has_flow = has_flow or ("flow_pred" in pred_data)
                        has_static = has_static or ("point_static" in pred_data)
                        has_point_loss = has_point_loss or ("pts_point_loss" in pred_data)
                if has_pred and has_flow and has_static and has_point_loss:
                    break
            except Exception:
                continue
        return has_pred, has_flow, has_static, has_point_loss

    @staticmethod
    def _pack_vectors(magic: bytes, vectors: np.ndarray) -> bytes:
        header = bytearray()
        header.extend(magic)
        header.extend(np.uint32(len(vectors)).tobytes())
        header.extend(np.uint32(0).tobytes())
        header.extend(np.uint32(0).tobytes())
        return b"".join(
            [
                bytes(header),
                np.ascontiguousarray(vectors, dtype=np.float32).tobytes(),
            ]
        )

    @staticmethod
    def _pack_labels(magic: bytes, labels: np.ndarray) -> bytes:
        header = bytearray()
        header.extend(magic)
        header.extend(np.uint32(len(labels)).tobytes())
        header.extend(np.uint32(0).tobytes())
        header.extend(np.uint32(0).tobytes())
        return b"".join(
            [
                bytes(header),
                np.ascontiguousarray(labels, dtype=np.uint8).tobytes(),
            ]
        )

    @staticmethod
    def _pack_scalars(magic: bytes, values: np.ndarray) -> bytes:
        header = bytearray()
        header.extend(magic)
        header.extend(np.uint32(len(values)).tobytes())
        header.extend(np.uint32(0).tobytes())
        header.extend(np.uint32(0).tobytes())
        return b"".join(
            [
                bytes(header),
                np.ascontiguousarray(values, dtype=np.float32).tobytes(),
            ]
        )

    @lru_cache(maxsize=16)
    def load_frame_images_meta(self, index: int) -> list[dict[str, str]]:
        return [
            {
                "cameraKey": source["cameraKey"],
                "cameraLabel": source["cameraLabel"],
            }
            for source in extract_frame_image_sources(self._load_frame_info(index))
        ]

    @lru_cache(maxsize=16)
    def load_frame_image(self, index: int, camera_key: str) -> tuple[bytes, str] | None:
        frame_info = self._load_frame_info(index)
        source = resolve_frame_image_source(frame_info, camera_key)
        if source is None:
            return None
        from ad_cloud.adrn.data_seeker.utils import bytes_to_image_array

        image_bytes = _load_image_bytes_for_adrn(source["adrn"])
        if image_bytes is None:
            return None
        image_array = normalize_image_array_for_browser(bytes_to_image_array(image_bytes))
        if image_array.size == 0:
            return None
        pil_image = Image.fromarray(image_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"

    @lru_cache(maxsize=2)
    def _load_bundle(self, index: int) -> dict[str, Any]:
        frame_info_path = self._frame_info_path(index)
        external_bundle = load_frame_bundle_for_source(
            frame_info_path,
            eval_dir=self.eval_dir,
            at720=self.at720,
        )
        if external_bundle is not None:
            return self._normalize_external_bundle(frame_info_path, external_bundle)

        data_info = self._load_frame_info(index)

        lidar_adrn = data_info["adrns"] if "adrns" in data_info else data_info.get("adrn")
        lidar_data, base_mask = load_points_for_frame(lidar_adrn, at720=self.at720)
        if lidar_data is None or base_mask is None:
            raise ValueError(f"Failed to load lidar points for {frame_info_path.name}")

        pred_data = None
        pred_path = self._prediction_path(data_info)
        if pred_path is not None and pred_path.exists():
            with open(pred_path, "rb") as handle:
                pred_data = pickle.load(handle)

        point_selector = base_mask
        if pred_data and pred_data.get("points_voxel_sample_mask") is not None:
            point_selector = np.asarray(pred_data["points_voxel_sample_mask"])
            expected_points = int(pred_data.get("current_pts_num", lidar_data.shape[0]))
            if expected_points != lidar_data.shape[0]:
                raise ValueError(
                    f"current_pts_num mismatch for {frame_info_path.name}: {expected_points} vs {lidar_data.shape[0]}"
                )

        selected_points = lidar_data[point_selector]
        original_point_count = int(lidar_data.shape[0])
        voxel_selected_point_count = int(selected_points.shape[0])
        range_mask = self._point_range_mask(selected_points)
        visible_points = selected_points[range_mask]
        in_range_point_count = int(visible_points.shape[0])

        gt_seg_raw = data_info.get("gt_seg")
        raw_gt_visible = None
        if gt_seg_raw is None:
            gt_labels = np.zeros((len(visible_points),), dtype=np.int32)
        else:
            raw_gt_visible = np.asarray(gt_seg_raw)[point_selector][range_mask]
            gt_labels = self._map_raw_gt_labels(raw_gt_visible)
        ignore_mask = (gt_labels == 13).astype(np.uint8, copy=False)

        pred_count = None
        pred_labels = None
        pred_flow_values = None
        gt_flow_values = self._load_gt_flow_values(data_info, point_selector, range_mask)
        gt_static_labels = self._load_gt_static_labels(data_info, point_selector, range_mask)
        pred_static_labels = None
        point_loss = None
        if pred_data and pred_data.get("pts_results") is not None:
            pred_count = pred_data.get("counts")
            pred_labels = self._align_pred_array(
                self._normalize_label(pred_data["pts_results"]),
                pred_count,
                len(selected_points),
                range_mask,
                "pts_results",
            )
            pred_flow_values = self._align_pred_array(
                pred_data.get("flow_pred"),
                pred_count,
                len(selected_points),
                range_mask,
                "flow_pred",
            )
            pred_static_labels = self._align_pred_array(
                pred_data.get("point_static"),
                pred_count,
                len(selected_points),
                range_mask,
                "point_static",
            )
            point_loss = self._align_pred_array(
                pred_data.get("pts_point_loss"),
                pred_count,
                len(selected_points),
                range_mask,
                "pts_point_loss",
            )

        positions = np.ascontiguousarray(visible_points[:, :3], dtype=np.float32)
        gt_labels = np.ascontiguousarray(gt_labels.astype(np.int32, copy=False))
        pred_labels = (
            None
            if pred_labels is None
            else np.ascontiguousarray(pred_labels.astype(np.int32, copy=False))
        )
        pred_flow_values = (
            None
            if pred_flow_values is None
            else np.ascontiguousarray(pred_flow_values[:, :3], dtype=np.float32)
        )
        gt_flow_values = (
            None
            if gt_flow_values is None
            else np.ascontiguousarray(gt_flow_values[:, :3], dtype=np.float32)
        )
        pred_static_labels = (
            None
            if pred_static_labels is None
            else np.ascontiguousarray(np.clip(pred_static_labels.astype(np.int32), 0, 1).astype(np.uint8))
        )
        point_loss = (
            None
            if point_loss is None
            else np.ascontiguousarray(np.asarray(point_loss, dtype=np.float32).reshape(-1))
        )
        ignore_mask = np.ascontiguousarray(ignore_mask.astype(np.uint8, copy=False))
        positions, gt_labels, pred_labels, flow_values, pred_static_labels, point_loss, ignore_mask = self._apply_sample_rate(
            positions,
            gt_labels,
            pred_labels,
            pred_flow_values,
            pred_static_labels,
            point_loss,
            ignore_mask,
            self.sample_rate,
        )
        if gt_flow_values is not None and self.sample_rate > 1:
            gt_flow_values = gt_flow_values[:: self.sample_rate]
        if gt_static_labels is not None and self.sample_rate > 1:
            gt_static_labels = gt_static_labels[:: self.sample_rate]
        ignored_point_count_after_downsample = int(ignore_mask.sum()) if ignore_mask is not None else 0
        point_loss_valid_mask = None if ignore_mask is None else (ignore_mask == 0)
        point_loss_range = self._compute_scalar_range(point_loss, point_loss_valid_mask)
        gt_detections = self._build_gt_detections(data_info)
        gt_next_detections = self._build_next_gt_detections(data_info)
        pred_detections = self._build_pred_detections(pred_data)
        default_static_labels = gt_static_labels
        if self.label_source == "pred" and pred_static_labels is not None:
            default_static_labels = pred_static_labels
        return {
            "name": frame_info_path.name,
            "positions": positions,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
            "gt_detections": gt_detections,
            "gt_next_detections": gt_next_detections,
            "pred_detections": pred_detections,
            "pred_flow": flow_values,
            "gt_flow": gt_flow_values,
            "pred_static_labels": pred_static_labels,
            "gt_static_labels": gt_static_labels,
            "point_loss": point_loss,
            "ignore_mask": ignore_mask,
            "point_loss_range": point_loss_range,
            "log_info": self._build_frame_log_info(
                frame_info_path=frame_info_path,
                data_info=data_info,
                pred_path=pred_path,
                raw_gt_visible=raw_gt_visible,
                mapped_gt_visible=gt_labels,
                visible_point_count=len(positions),
                original_point_count=original_point_count,
                voxel_selected_point_count=voxel_selected_point_count,
                in_range_point_count=in_range_point_count,
                ignored_point_count_after_downsample=ignored_point_count_after_downsample,
                gt_det_count=len(gt_detections) + len(gt_next_detections),
                pred_det_count=len(pred_detections),
                flow_values=flow_values if flow_values is not None else gt_flow_values,
                static_labels=default_static_labels,
                point_loss_range=point_loss_range,
            ),
        }

    @staticmethod
    def _pack_frame(name: str, positions: np.ndarray, colors: np.ndarray) -> bytes:
        name_bytes = name.encode("utf-8")
        pad_len = (-len(name_bytes)) % 4
        header = bytearray()
        header.extend(b"PCD0")
        header.extend(np.uint32(len(positions)).tobytes())
        header.extend(np.uint32(len(name_bytes)).tobytes())
        header.extend(np.uint32(0).tobytes())
        return b"".join(
            [
                bytes(header),
                name_bytes,
                b"\x00" * pad_len,
                np.ascontiguousarray(positions, dtype=np.float32).tobytes(),
                np.ascontiguousarray(colors, dtype=np.uint8).tobytes(),
            ]
        )

    @staticmethod
    def _normalize_label_source(label_source: str | None, default: str) -> str:
        if label_source in {"gt", "pred"}:
            return label_source
        return default

    @lru_cache(maxsize=16)
    def load_frame(self, index: int, label_source: str | None = None) -> bytes:
        bundle = self._load_bundle(index)
        label_source = self._normalize_label_source(label_source, self.label_source)
        labels = bundle["gt_labels"]
        if label_source == "pred" and bundle["pred_labels"] is not None:
            labels = bundle["pred_labels"]
        colors = np.ascontiguousarray(RENDER_CLASSNAME_TO_COLOR[labels], dtype=np.uint8)
        return self._pack_frame(bundle["name"], bundle["positions"], colors)

    @lru_cache(maxsize=8)
    def load_previous_points(self, index: int) -> bytes | None:
        data_info = self._load_frame_info(index)
        sweeps = data_info.get("sweeps")
        if not isinstance(sweeps, list) or not sweeps:
            return None
        ego2global = data_info.get("ego2global")
        if ego2global is None:
            return None
        ego2global = self._normalize_pose_matrix(ego2global, "ego2global")

        previous_positions: list[np.ndarray] = []
        for sweep in sweeps:
            if not isinstance(sweep, dict):
                continue
            sweep_pose = sweep.get("ego2global")
            if sweep_pose is None:
                continue
            lidar_adrn = sweep.get("adrns") if "adrns" in sweep else sweep.get("adrn")
            if not lidar_adrn:
                continue
            sweep_lidar, sweep_mask = load_points_for_frame(lidar_adrn, at720=self.at720)
            if sweep_lidar is None or sweep_mask is None:
                continue
            selected_points = np.asarray(sweep_lidar)[sweep_mask]
            if selected_points.size == 0:
                continue
            transformed = self._transform_points_to_current_frame(
                np.asarray(selected_points[:, :3], dtype=np.float32),
                ego2global,
                self._normalize_pose_matrix(sweep_pose, "sweeps.ego2global"),
            )
            range_mask = self._point_range_mask(transformed)
            transformed = transformed[range_mask]
            if transformed.size == 0:
                continue
            previous_positions.append(transformed)

        if not previous_positions:
            return None
        positions = np.ascontiguousarray(np.concatenate(previous_positions, axis=0), dtype=np.float32)
        if self.sample_rate > 1:
            positions = positions[:: self.sample_rate]
        if positions.size == 0:
            return None
        colors = np.full((positions.shape[0], 3), 160, dtype=np.uint8)
        return self._pack_frame(f"{self._frame_info_path(index).name}:prev", positions, colors)

    @lru_cache(maxsize=16)
    def load_detections(self, index: int, label_source: str | None = None) -> list[dict]:
        bundle = self._load_bundle(index)
        label_source = self._normalize_label_source(label_source, self.label_source)
        if label_source == "pred" and bundle["pred_detections"]:
            return bundle["pred_detections"]
        if label_source == "gt":
            return bundle["gt_detections"] + bundle["gt_next_detections"]
        return bundle["gt_detections"]

    @lru_cache(maxsize=16)
    def load_frame_log_info(self, index: int, label_source: str | None = None) -> dict[str, Any]:
        bundle = self._load_bundle(index)
        label_source = self._normalize_label_source(label_source, self.label_source)
        log_info = {
            "entries": [dict(item) for item in bundle["log_info"]["entries"]]
        }
        log_info["entries"].insert(0, {"key": "mode_current", "value": label_source})
        return log_info

    @lru_cache(maxsize=8)
    def load_flow(self, index: int, label_source: str | None = None) -> bytes | None:
        bundle = self._load_bundle(index)
        label_source = self._normalize_label_source(label_source, self.label_source)
        flow = bundle["pred_flow"] if label_source == "pred" else bundle["gt_flow"]
        if flow is None:
            return None
        return self._pack_vectors(b"FLW0", flow[:, :3])

    @lru_cache(maxsize=8)
    def load_static_labels(self, index: int, label_source: str | None = None) -> bytes | None:
        bundle = self._load_bundle(index)
        label_source = self._normalize_label_source(label_source, self.label_source)
        static_labels = bundle["pred_static_labels"] if label_source == "pred" else bundle["gt_static_labels"]
        if static_labels is None:
            return None
        return self._pack_labels(b"STA0", static_labels)

    @lru_cache(maxsize=8)
    def load_sub_box_flows(self, index: int, label_source: str | None = None) -> list[dict[str, Any]]:
        label_source = self._normalize_label_source(label_source, self.label_source)
        if label_source != "gt":
            return []
        data_info = self._load_frame_info(index)
        flow_gt = data_info.get("flow_gt")
        split_box_masks_by_box = data_info.get("is_in_split_box")
        split_box_centers_by_box = data_info.get("main_id_center")
        if flow_gt is None or not isinstance(split_box_masks_by_box, dict) or split_box_centers_by_box is None:
            return []

        flow_gt = self._ensure_2d_array(flow_gt, "flow_gt", 3).astype(np.float32, copy=False)
        raw_mask = self._get_gt_flow_mask(data_info)
        if raw_mask is not None:
            flow_mask = np.asarray(raw_mask).reshape(-1).astype(bool, copy=False)
            if flow_mask.shape[0] == flow_gt.shape[0]:
                flow_gt = np.array(flow_gt, copy=True)
                flow_gt[~flow_mask] = 0.0
        items: list[dict[str, Any]] = []
        for box_id, split_box_masks in split_box_masks_by_box.items():
            try:
                split_box_centers = split_box_centers_by_box[box_id]
            except Exception:
                continue
            split_box_centers = np.asarray(split_box_centers, dtype=np.float32)
            if split_box_centers.ndim == 1:
                split_box_centers = np.expand_dims(split_box_centers, axis=0)
            if split_box_centers.ndim != 2 or split_box_centers.shape[1] < 3:
                continue

            for sub_box_index, split_box_mask in enumerate(split_box_masks):
                if sub_box_index >= split_box_centers.shape[0]:
                    break
                bool_split_box_mask = np.asarray(split_box_mask).reshape(-1).astype(bool, copy=False)
                if bool_split_box_mask.shape[0] != flow_gt.shape[0]:
                    continue
                split_box_flow = flow_gt[bool_split_box_mask, :3]
                if split_box_flow.shape[0] == 0:
                    continue
                nonzero_split_box_flow = split_box_flow[np.linalg.norm(split_box_flow[:, :2], axis=1) > 0.0]
                if nonzero_split_box_flow.shape[0] == 0:
                    continue
                unique_flow = np.unique(np.ascontiguousarray(nonzero_split_box_flow, dtype=np.float32), axis=0)
                if unique_flow.shape[0] != 1:
                    continue
                flow = unique_flow[0]
                xy_norm = float(np.linalg.norm(flow[:2]))
                if not np.isfinite(xy_norm) or xy_norm <= 0.0:
                    continue
                center = split_box_centers[sub_box_index, :3]
                xy_extent = (
                    float(np.linalg.norm(np.ptp(split_box_centers[:, :2], axis=0)))
                    if split_box_centers.shape[0] > 1
                    else 0.0
                )
                offset_step = min(0.04, max(0.01, xy_extent * 0.03))
                height_offset = float(sub_box_index) * offset_step
                items.append(
                    {
                        "boxId": int(box_id),
                        "subBoxIndex": int(sub_box_index),
                        "center": [float(center[0]), float(center[1]), float(center[2])],
                        "flow": [float(flow[0]), float(flow[1]), float(flow[2])],
                        "xyNorm": xy_norm,
                        "heightOffset": height_offset,
                    }
                )
        return items

    @lru_cache(maxsize=8)
    def load_point_loss(self, index: int) -> bytes | None:
        point_loss = self._load_bundle(index)["point_loss"]
        if point_loss is None:
            return None
        return self._pack_scalars(b"LOS0", point_loss)

    @lru_cache(maxsize=8)
    def load_ignore_mask(self, index: int) -> bytes:
        ignore_mask = self._load_bundle(index)["ignore_mask"]
        return self._pack_labels(b"IGN0", ignore_mask)

    def meta(self, fps: float, point_size: float) -> dict:
        return {
            "frameCount": len(self.frame_paths),
            "fps": fps,
            "pointSize": point_size,
            "sampleRate": self.sample_rate,
            "labelSource": self.label_source,
            "at720": self.at720,
            "hasEvalResults": self.has_pred_modality or self.eval_dir is not None,
            "hasFlow": self.has_flow_modality,
            "hasStaticLabels": self.has_static_modality,
            "hasPointLoss": self.has_point_loss_modality,
            "classes": [
                {
                    "id": class_id,
                    "name": class_name,
                    "color": CLASSNAME_TO_COLOR[class_id].tolist(),
                }
                for class_id, class_name in PTS_SEG_ID_AND_CLS.items()
            ],
            "pklFile": str(self.pkl_file),
            "evalDir": "" if self.eval_dir is None else str(self.eval_dir),
        }


class ViewerState:
    def __init__(self, store: FrameStore):
        self._store = store
        self._lock = threading.RLock()

    def get_store(self) -> FrameStore:
        with self._lock:
            return self._store

    def open_store(
        self,
        pkl_file: str,
        eval_dir: str,
        sample_rate: int,
        label_source: str,
        at720: bool,
    ) -> FrameStore:
        store = FrameStore(
            pkl_file=pkl_file,
            eval_dir=eval_dir,
            sample_rate=sample_rate,
            label_source=label_source,
            at720=at720,
        )
        with self._lock:
            self._store = store
            return self._store

    @staticmethod
    def suggest_paths(
        prefix: str,
        kind: str,
        limit: int = 40,
    ) -> list[dict[str, str | bool]]:
        raw_prefix = (prefix or "").strip()
        expanded = Path(raw_prefix).expanduser() if raw_prefix else Path.home()
        sep = "/" if "/" in raw_prefix or raw_prefix.startswith("~") else "/"

        if raw_prefix.endswith(sep):
            base_dir = expanded
            partial = ""
        elif expanded.exists() and expanded.is_dir():
            base_dir = expanded
            partial = ""
        else:
            base_dir = expanded.parent
            partial = expanded.name

        if not base_dir.exists() or not base_dir.is_dir():
            return []

        suggestions: list[dict[str, str | bool]] = []
        partial_lower = partial.lower()
        for child in sorted(base_dir.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            name_lower = child.name.lower()
            if partial_lower and not name_lower.startswith(partial_lower):
                continue
            if kind == "eval_dir" and not child.is_dir():
                continue
            if kind == "pkl_file" and not (child.is_dir() or child.suffix == ".pkl"):
                continue
            display_path = str(child)
            if child.is_dir():
                display_path = display_path.rstrip("/") + "/"
            suggestions.append(
                {
                    "path": display_path,
                    "isDir": child.is_dir(),
                }
            )
            if len(suggestions) >= limit:
                break
        return suggestions

    @staticmethod
    def list_path_entries(
        path_value: str,
        kind: str,
        limit: int = 120,
    ) -> dict[str, object]:
        raw_value = (path_value or "").strip()
        expanded = Path(raw_value).expanduser() if raw_value else Path.home()

        if raw_value.endswith("/") or (expanded.exists() and expanded.is_dir()):
            base_dir = expanded
        else:
            base_dir = expanded.parent

        if not base_dir.exists() or not base_dir.is_dir():
            return {"directory": str(base_dir), "entries": []}

        entries: list[dict[str, str | bool]] = []
        parent_dir = base_dir.parent
        if parent_dir != base_dir:
            entries.append(
                {
                    "name": "../",
                    "path": str(parent_dir).rstrip("/") + "/",
                    "isDir": True,
                }
            )

        for child in sorted(base_dir.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if kind == "eval_dir" and not child.is_dir():
                continue
            if kind == "pkl_file" and not (child.is_dir() or child.suffix == ".pkl"):
                continue
            entries.append(
                {
                    "name": f"{child.name}/" if child.is_dir() else child.name,
                    "path": str(child).rstrip("/") + ("/" if child.is_dir() else ""),
                    "isDir": child.is_dir(),
                }
            )
            if len(entries) >= limit:
                break

        return {"directory": str(base_dir), "entries": entries}


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Point Cloud Browser Viewer</title>
  <style>
    :root {
      --bg: #0d1117;
      --panel: rgba(15, 23, 42, 0.88);
      --panel-border: rgba(148, 163, 184, 0.24);
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #f59e0b;
      --sidebar-width: 340px;
    }
    html, body {
      margin: 0;
      height: 100%;
      background:
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.14), transparent 26%),
        radial-gradient(circle at bottom left, rgba(245, 158, 11, 0.12), transparent 30%),
        var(--bg);
      color: var(--text);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      overflow: hidden;
    }
    #app {
      height: 100%;
      display: grid;
      grid-template-columns: minmax(280px, var(--sidebar-width)) 10px minmax(0, 1fr);
    }
    #sidebar {
      background: var(--panel);
      border-right: 1px solid var(--panel-border);
      backdrop-filter: blur(14px);
      padding: 18px 18px 12px;
      overflow: auto;
      box-sizing: border-box;
    }
    #sidebarHeader {
      margin: 0 0 14px;
    }
    #sidebarSplitter {
      position: relative;
      cursor: col-resize;
      background: linear-gradient(
        180deg,
        rgba(148, 163, 184, 0.05),
        rgba(148, 163, 184, 0.18),
        rgba(148, 163, 184, 0.05)
      );
    }
    #sidebarSplitter::after {
      content: "";
      position: absolute;
      left: 3px;
      top: 0;
      bottom: 0;
      width: 4px;
      background: rgba(148, 163, 184, 0.08);
    }
    #viewer {
      position: relative;
      min-width: 0;
      height: 100%;
    }
    #canvasWrap {
      position: absolute;
      inset: 0;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 20px;
      letter-spacing: 0.02em;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 16px;
      line-height: 1.45;
    }
    .block {
      margin-bottom: 16px;
      padding: 12px;
      border: 1px solid rgba(148, 163, 184, 0.34);
      border-radius: 14px;
      background:
        linear-gradient(180deg, rgba(30, 41, 59, 0.72), rgba(15, 23, 42, 0.58));
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.04),
        0 10px 28px rgba(2, 6, 23, 0.2);
    }
    .sidebarPanelHeader {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0;
      border: 0;
      background: transparent;
      color: inherit;
      cursor: pointer;
      text-align: left;
    }
    .sidebarPanelHeader .sectionTitle {
      margin: 0;
    }
    .sidebarPanelHeader:hover .sectionTitle,
    .sidebarPanelHeader:focus-visible .sectionTitle {
      color: #f8fafc;
    }
    .sidebarPanelHeader:focus-visible {
      outline: 2px solid rgba(56, 189, 248, 0.7);
      outline-offset: 4px;
      border-radius: 8px;
    }
    .sidebarPanelChevron {
      flex: 0 0 auto;
      color: var(--muted);
      font-size: 14px;
      transition: transform 0.18s ease, color 0.18s ease;
    }
    .block.is-collapsed .sidebarPanelChevron {
      transform: rotate(-90deg);
    }
    .sidebarPanelBody {
      padding-top: 10px;
    }
    .block.is-collapsed .sidebarPanelBody {
      display: none;
    }
    .sectionTitle {
      margin: 0 0 10px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #cbd5e1;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }
    .row:last-child {
      margin-bottom: 0;
    }
    button {
      border: 0;
      border-radius: 10px;
      background: linear-gradient(135deg, var(--accent), #0ea5e9);
      color: #06131c;
      font-weight: 700;
      padding: 8px 12px;
      cursor: pointer;
    }
    button.secondary {
      background: rgba(148, 163, 184, 0.18);
      color: var(--text);
      font-weight: 600;
    }
    input[type="text"], select, textarea {
      width: 100%;
      border: 1px solid rgba(148, 163, 184, 0.24);
      border-radius: 10px;
      background: rgba(2, 8, 23, 0.75);
      color: var(--text);
      padding: 9px 11px;
      font: inherit;
      box-sizing: border-box;
    }
    .pathInput {
      min-height: 84px;
      resize: vertical;
      font-size: 11px;
      line-height: 1.35;
      font-family: "IBM Plex Mono", monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .pathBrowser {
      margin-top: 6px;
      padding: 8px;
      border-radius: 10px;
      background: rgba(2, 8, 23, 0.42);
      border: 1px solid rgba(148, 163, 184, 0.14);
      display: grid;
      gap: 6px;
    }
    .pathBrowserLabel {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 10px;
    }
    .pathEntries {
      min-height: 112px;
      max-height: 180px;
      padding: 6px 8px;
      font-size: 11px;
      font-family: "IBM Plex Mono", monospace;
    }
    .pathEntries.empty {
      color: rgba(148, 163, 184, 0.7);
    }
    input[type="text"]::placeholder {
      color: rgba(148, 163, 184, 0.7);
    }
    label {
      display: block;
      width: 100%;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .value {
      color: var(--text);
      font-size: 13px;
      min-width: 52px;
      text-align: right;
    }
    #status {
      font-size: 12px;
      line-height: 1.5;
      color: var(--text);
      word-break: break-word;
      overflow-wrap: anywhere;
      display: grid;
      gap: 8px;
    }
    .statusRow {
      display: grid;
      gap: 4px;
    }
    .statusKey {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 11px;
    }
    .statusVal {
      color: var(--text);
      font-family: "IBM Plex Mono", monospace;
      font-size: 11px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
    }
    #legend {
      display: grid;
      gap: 6px;
    }
    .legendItem {
      display: flex;
      gap: 10px;
      align-items: center;
      font-size: 13px;
      color: var(--text);
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      flex: 0 0 auto;
      border: 1px solid rgba(255, 255, 255, 0.2);
    }
    #overlay {
      position: absolute;
      right: 18px;
      top: 18px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(2, 8, 23, 0.66);
      border: 1px solid rgba(148, 163, 184, 0.2);
      backdrop-filter: blur(8px);
      font-size: 13px;
      line-height: 1.45;
      min-width: 220px;
    }
    #modeBar {
      position: absolute;
      right: 18px;
      top: 122px;
      display: flex;
      gap: 6px;
      padding: 8px;
      border-radius: 12px;
      background: rgba(2, 8, 23, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.2);
      backdrop-filter: blur(8px);
      box-sizing: border-box;
    }
    .modeButton {
      border: 0;
      border-radius: 10px;
      background: rgba(148, 163, 184, 0.14);
      color: var(--text);
      font-weight: 700;
      padding: 8px 12px;
      cursor: pointer;
    }
    .modeButton.active {
      background: linear-gradient(135deg, var(--accent), #0ea5e9);
      color: #06131c;
    }
    .modeButton:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    #lossHistogramPanel {
      position: absolute;
      left: 18px;
      top: 18px;
      width: min(36vw, 420px);
      min-width: 280px;
      border-radius: 14px;
      background: rgba(2, 8, 23, 0.86);
      border: 1px solid rgba(148, 163, 184, 0.24);
      backdrop-filter: blur(10px);
      overflow: hidden;
      display: none;
      box-sizing: border-box;
    }
    #lossHistogramPanel.open {
      display: block;
    }
    #imageFloatingPanels {
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 8;
    }
    #imageFloatingPanel,
    .imageFloatingPanel {
      position: absolute;
      left: 32px;
      top: 150px;
      width: 520px;
      height: 320px;
      border-radius: 14px;
      background: rgba(2, 8, 23, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.24);
      backdrop-filter: blur(10px);
      overflow: hidden;
      display: none;
      box-sizing: border-box;
      pointer-events: auto;
    }
    #imageFloatingPanel.open,
    .imageFloatingPanel.open {
      display: flex;
      flex-direction: column;
    }
    #imageFloatingPanelHeader,
    .imageFloatingPanelHeader {
      display: flex;
      align-items: center;
      gap: 10px;
      justify-content: space-between;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.16);
      cursor: move;
      user-select: none;
    }
    #imageFloatingPanelTitle,
    .imageFloatingPanelTitle {
      font-size: 13px;
      font-weight: 700;
    }
    #imageFloatingPanelControls,
    .imageFloatingPanelControls {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      flex: 1 1 auto;
      justify-content: flex-end;
    }
    #imageFloatingPanelBody,
    .imageFloatingPanelBody {
      position: relative;
      flex: 1 1 auto;
      min-height: 0;
      background: rgba(15, 23, 42, 0.5);
    }
    #imageFloatingPanelState,
    .imageFloatingPanelState {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 18px;
      text-align: center;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    #imageFloatingPanelState.is-hidden,
    .imageFloatingPanelState.is-hidden {
      display: none;
    }
    #imageFloatingPanelImg,
    .imageFloatingPanelImg {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: rgba(2, 8, 23, 0.75);
    }
    #imageFloatingPanelImg.is-hidden,
    .imageFloatingPanelImg.is-hidden {
      display: none;
    }
    #imageFloatingPanelResizeHandle,
    .imageFloatingPanelResizeHandle {
      position: absolute;
      right: 0;
      bottom: 0;
      width: 20px;
      height: 20px;
      cursor: nwse-resize;
      background:
        linear-gradient(135deg, transparent 0 45%, rgba(148, 163, 184, 0.6) 45% 52%, transparent 52% 62%, rgba(148, 163, 184, 0.45) 62% 69%, transparent 69%);
      z-index: 2;
    }
    #lossHistogramHeader {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.16);
      font-size: 13px;
      font-weight: 700;
    }
    #lossHistogramClose {
      border: 0;
      background: rgba(148, 163, 184, 0.14);
      color: var(--text);
      border-radius: 8px;
      width: 28px;
      height: 28px;
      padding: 0;
      font-size: 16px;
      line-height: 1;
    }
    #lossHistogramCanvas {
      display: block;
      width: 100%;
      height: 220px;
      background: rgba(2, 8, 23, 0.42);
    }
    #bboxPanel {
      position: absolute;
      right: 18px;
      top: 172px;
      width: 320px;
      min-width: 240px;
      min-height: 180px;
      max-width: min(50vw, 560px);
      max-height: min(70vh, 640px);
      padding: 0;
      border-radius: 14px;
      background: rgba(2, 8, 23, 0.86);
      border: 1px solid rgba(148, 163, 184, 0.24);
      backdrop-filter: blur(10px);
      overflow: auto;
      resize: both;
      display: none;
      box-sizing: border-box;
    }
    #bboxPanel.open {
      display: block;
    }
    #bboxPanelHeader {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px 10px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.16);
      font-weight: 700;
      font-size: 14px;
    }
    #bboxPanelClose {
      border: 0;
      background: rgba(148, 163, 184, 0.14);
      color: var(--text);
      border-radius: 8px;
      width: 28px;
      height: 28px;
      padding: 0;
      font-size: 16px;
      line-height: 1;
    }
    #bboxPanelBody {
      padding: 12px 14px 14px;
      display: grid;
      gap: 10px;
      font-size: 13px;
      line-height: 1.5;
    }
    #logPanel {
      position: absolute;
      left: 18px;
      bottom: 18px;
      width: min(42vw, 520px);
      height: min(52vh, 520px);
      min-width: 280px;
      min-height: 120px;
      max-width: min(56vw, 760px);
      border-radius: 14px;
      background: rgba(2, 8, 23, 0.84);
      border: 1px solid rgba(148, 163, 184, 0.24);
      backdrop-filter: blur(10px);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-sizing: border-box;
      min-height: 0;
    }
    #logPanel.minimized {
      resize: none;
      min-height: 0;
      height: auto;
    }
    #logPanelHeader {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.16);
      font-size: 13px;
      font-weight: 700;
      cursor: move;
      user-select: none;
    }
    #logPanelControls {
      display: flex;
      gap: 8px;
    }
    #logPanelHeader button {
      border: 0;
      background: rgba(148, 163, 184, 0.14);
      color: var(--text);
      border-radius: 8px;
      width: 28px;
      height: 28px;
      padding: 0;
      font-size: 14px;
      line-height: 1;
    }
    #logPanelBody {
      padding: 10px 12px 12px;
      font-size: 12px;
      line-height: 1.45;
      display: flex;
      flex-direction: column;
      gap: 10px;
      flex: 1 1 auto;
      min-height: 0;
      overflow: hidden;
    }
    #logPanel.minimized #logPanelBody {
      display: none;
    }
    .logSection {
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-height: 0;
    }
    #frameLogSection {
      flex: 0 0 auto;
    }
    #eventLogSection {
      flex: 1 1 auto;
      min-height: 160px;
    }
    #eventLogViewport {
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
      padding-right: 4px;
    }
    .logSectionTitle {
      color: #93c5fd;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }
    .logEntry {
      white-space: pre-wrap;
      word-break: break-word;
      color: #dbe7f4;
      font-family: "IBM Plex Mono", monospace;
    }
    .logEntry.error {
      color: #fda4af;
    }
    .logEntry.warn {
      color: #fcd34d;
    }
    .logKeyVal {
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 10px;
      align-items: start;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .logKey {
      color: #93c5fd;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .logVal {
      color: #dbe7f4;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .logResizeHandle {
      position: absolute;
      z-index: 2;
    }
    .logResizeHandle.n,
    .logResizeHandle.s {
      left: 0;
      right: 0;
      height: 18px;
      cursor: ns-resize;
    }
    .logResizeHandle.e,
    .logResizeHandle.w {
      top: 0;
      bottom: 0;
      width: 14px;
      cursor: ew-resize;
    }
    .logResizeHandle.n { top: 0; }
    .logResizeHandle.s { bottom: 0; }
    .logResizeHandle.e { right: 0; }
    .logResizeHandle.w { left: 0; }
    .logResizeHandle.nw,
    .logResizeHandle.ne,
    .logResizeHandle.sw,
    .logResizeHandle.se {
      width: 16px;
      height: 16px;
    }
    .logResizeHandle.nw { left: 0; top: 0; cursor: nwse-resize; }
    .logResizeHandle.ne { right: 0; top: 0; cursor: nesw-resize; }
    .logResizeHandle.sw { left: 0; bottom: 0; cursor: nesw-resize; }
    .logResizeHandle.se { right: 0; bottom: 0; cursor: nwse-resize; }
    .bboxRow {
      display: grid;
      grid-template-columns: 96px 1fr;
      gap: 10px;
      align-items: start;
    }
    .bboxKey {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 11px;
      padding-top: 2px;
    }
    .bboxVal {
      color: var(--text);
      font-family: "IBM Plex Mono", monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .kbd {
      border-radius: 6px;
      border: 1px solid rgba(148, 163, 184, 0.26);
      padding: 2px 6px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      color: var(--muted);
    }
    a {
      color: var(--accent);
    }
  </style>
</head>
<body>
  <div id="app">
    <aside id="sidebar">
      <div id="sidebarHeader">
        <h1>Point Cloud Viewer</h1>
        <div class="sub">
          Browser player for lazy-loaded PKL frame datasets.
          Use <span class="kbd">Space</span> to pause or resume.
        </div>
      </div>

      <div class="block is-collapsed" data-panel-title="Data Source">
        <button class="sidebarPanelHeader sidebarPanelToggle" type="button" aria-expanded="false">
          <div class="sectionTitle">Data Source</div>
          <div class="sidebarPanelChevron">▾</div>
        </button>
        <div class="sidebarPanelBody">
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="rootDirInput">PKL File</label>
              <textarea id="rootDirInput" class="pathInput" spellcheck="false" placeholder="/path/to/val.pkl"></textarea>
              <div class="pathBrowser">
                <div class="pathBrowserLabel">Files / Folders In Current Directory</div>
                <select id="pklPathEntries" class="pathEntries" size="6"></select>
              </div>
            </div>
          </div>
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="subdirSelect">Eval Directory</label>
              <textarea id="subdirSelect" class="pathInput" spellcheck="false" placeholder="/path/to/eval_dir (optional)"></textarea>
              <div class="pathBrowser">
                <div class="pathBrowserLabel">Folders In Current Directory</div>
                <select id="evalDirEntries" class="pathEntries" size="6"></select>
              </div>
            </div>
          </div>
          <div class="row">
            <button class="secondary" id="scanRoot">Load Dataset</button>
            <button id="loadSelected">Reload Dataset</button>
          </div>
        </div>
      </div>

      <div class="block is-collapsed" data-panel-title="Playback">
        <button class="sidebarPanelHeader sidebarPanelToggle" type="button" aria-expanded="false">
          <div class="sectionTitle">Playback</div>
          <div class="sidebarPanelChevron">▾</div>
        </button>
        <div class="sidebarPanelBody">
          <div class="row">
            <button id="playPause">Pause</button>
            <button class="secondary" id="prevFrame">Prev</button>
            <button class="secondary" id="nextFrame">Next</button>
            <button class="secondary" id="togglePrevPoints">Prev Pts: Off</button>
          </div>
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="frameSlider">Frame</label>
              <input id="frameSlider" type="range" min="0" max="0" value="0" step="1" />
            </div>
            <div class="value" id="frameValue">0 / 0</div>
          </div>
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="frameJumpInput">Frame ID</label>
              <input id="frameJumpInput" type="text" value="0" />
            </div>
            <button class="secondary" id="jumpFrame">Jump</button>
          </div>
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="framePercentInput">Progress %</label>
              <input id="framePercentInput" type="text" value="0.00" />
            </div>
            <button class="secondary" id="jumpPercent">Jump %</button>
          </div>
          <div class="row">
            <button class="secondary" id="jumpRandom">Random</button>
          </div>
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="fpsSlider">FPS</label>
              <input id="fpsSlider" type="range" min="1" max="20" value="5" step="1" />
            </div>
            <div class="value" id="fpsValue">5</div>
          </div>
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="pointSizeSlider">Point Size</label>
              <input id="pointSizeSlider" type="range" min="1" max="8" value="2" step="0.5" />
            </div>
            <div class="value" id="pointSizeValue">2.0</div>
          </div>
          <div class="row">
            <button class="secondary" id="resetView">Reset View</button>
            <button class="secondary" id="toggleAutoFit">Auto Fit: Off</button>
          </div>
        </div>
      </div>

      <div class="block is-collapsed" data-panel-title="OD Boxes">
        <button class="sidebarPanelHeader sidebarPanelToggle" type="button" aria-expanded="false">
          <div class="sectionTitle">OD Boxes</div>
          <div class="sidebarPanelChevron">▾</div>
        </button>
        <div class="sidebarPanelBody">
          <div class="row">
            <button class="secondary" id="toggleDetections">OD Boxes: On</button>
            <button class="secondary" id="toggleFutureBoxes">Future Boxes: On</button>
          </div>
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="detLineThicknessInput">OD Box Thickness</label>
              <input id="detLineThicknessInput" type="range" min="0.02" max="1.00" step="0.01" value="0.12" />
            </div>
            <div class="value" id="detLineThicknessValue">0.12</div>
          </div>
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="detScoreSlider">OD Score Filter</label>
              <input id="detScoreSlider" type="range" min="0" max="1" value="0" step="0.01" />
            </div>
            <div class="value" id="detScoreValue">0.00</div>
          </div>
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="detScoreInput">OD Score Input</label>
              <input id="detScoreInput" type="text" value="0.00" />
            </div>
          </div>
        </div>
      </div>

      <div class="block is-collapsed" data-panel-title="Motion / Static">
        <button class="sidebarPanelHeader sidebarPanelToggle" type="button" aria-expanded="false">
          <div class="sectionTitle">Motion / Static</div>
          <div class="sidebarPanelChevron">▾</div>
        </button>
        <div class="sidebarPanelBody">
          <div class="row">
            <button class="secondary" id="toggleFlow">Flow: Off</button>
            <button class="secondary" id="toggleSubBoxFlow">Sub-box Flow: Off</button>
          </div>
          <div class="row">
            <button class="secondary" id="toggleSubBoxFlowText">Flow Text: Off</button>
          </div>
          <div class="row">
            <button class="secondary" id="toggleStaticLabels">Static Labels: Off</button>
          </div>
        </div>
      </div>

      <div class="block is-collapsed" id="lossBlock" data-panel-title="Point Loss">
        <button class="sidebarPanelHeader sidebarPanelToggle" type="button" aria-expanded="false">
          <div class="sectionTitle">Point Loss</div>
          <div class="sidebarPanelChevron">▾</div>
        </button>
        <div class="sidebarPanelBody">
          <div id="lossControls">
            <div class="row">
              <button class="secondary" id="toggleHideLossBelowThreshold">Hide Below Threshold: Off</button>
              <button class="secondary" id="toggleHideIgnorePoints">Hide Ignore Points: Off</button>
            </div>
            <div class="row">
              <div style="flex: 1 1 100%;">
                <label for="lossRatioSlider">Top Loss Ratio %</label>
                <input id="lossRatioSlider" type="range" min="0" max="100" value="75" step="1" />
              </div>
              <div class="value" id="lossRatioValue">75.0%</div>
            </div>
            <div class="row">
              <div style="flex: 1 1 100%;">
                <label for="lossRatioInput">Top Loss Ratio Input</label>
                <input id="lossRatioInput" type="text" value="75.0" />
              </div>
            </div>
            <div class="row">
              <div style="flex: 1 1 100%;">
                <label for="lossThresholdSlider">Loss Threshold</label>
                <input id="lossThresholdSlider" type="range" min="0" max="1" value="0" step="any" />
              </div>
              <div class="value" id="lossThresholdValue">0.0000</div>
            </div>
            <div class="row">
              <div style="flex: 1 1 100%;">
                <label for="lossThresholdInput">Loss Threshold Input</label>
                <input id="lossThresholdInput" type="text" value="0.0000" />
              </div>
            </div>
            <div class="row">
              <div style="flex: 1 1 100%;">
                <label for="lossRangeInfo">Current Sample Loss Range</label>
                <input id="lossRangeInfo" type="text" value="N/A" readonly />
              </div>
            </div>
            <div class="row">
              <div style="flex: 1 1 100%;">
                <label for="lossAboveCount">Above Threshold Count</label>
                <input id="lossAboveCount" type="text" value="0" readonly />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="block is-collapsed" id="editBlock" data-panel-title="Edit Mode">
        <button class="sidebarPanelHeader sidebarPanelToggle" type="button" aria-expanded="false">
          <div class="sectionTitle">Edit Mode</div>
          <div class="sidebarPanelChevron">▾</div>
        </button>
        <div class="sidebarPanelBody">
          <div id="editControls">
            <div class="row">
              <button class="secondary" id="clearEditCurrent">Clear Current</button>
              <button id="saveEditRectangles">Save Rectangles</button>
            </div>
            <div class="row">
              <div style="flex: 1 1 100%;">
                <label for="editStatus">Edit Status</label>
                <input id="editStatus" type="text" value="EDIT mode off" readonly />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="block is-collapsed" data-panel-title="Image Panels">
        <button class="sidebarPanelHeader sidebarPanelToggle" type="button" aria-expanded="false">
          <div class="sectionTitle">Image Panels</div>
          <div class="sidebarPanelChevron">▾</div>
        </button>
        <div class="sidebarPanelBody">
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="imagePanelCameraSelect">Image Camera</label>
              <select id="imagePanelCameraSelect">
                <option value="">No image cameras</option>
              </select>
            </div>
          </div>
          <div class="row">
            <button id="addImagePanelWindow" type="button">Add Image Window</button>
            <button id="openImagePanel" type="button" hidden>Open Floating Panel</button>
          </div>
          <div class="row">
            <div style="flex: 1 1 100%;">
              <label for="imagePanelStatus">Image Panel Status</label>
              <input id="imagePanelStatus" type="text" value="No image cameras loaded." readonly />
            </div>
          </div>
        </div>
      </div>

      <div class="block is-collapsed" data-panel-title="Status">
        <button class="sidebarPanelHeader sidebarPanelToggle" type="button" aria-expanded="false">
          <div class="sectionTitle">Status</div>
          <div class="sidebarPanelChevron">▾</div>
        </button>
        <div class="sidebarPanelBody">
          <div id="status">Loading metadata...</div>
        </div>
      </div>

      <div class="block" data-panel-title="Legend">
        <button class="sidebarPanelHeader sidebarPanelToggle" type="button" aria-expanded="false">
          <div class="sectionTitle">Legend</div>
          <div class="sidebarPanelChevron">▾</div>
        </button>
        <div class="sidebarPanelBody">
          <div id="legend"></div>
        </div>
      </div>
    </aside>

    <div id="sidebarSplitter" aria-hidden="true"></div>

    <main id="viewer">
      <div id="canvasWrap"></div>
      <div id="overlay">
        <div id="overlayName">Frame: -</div>
        <div id="overlayPoints">Points: -</div>
        <div id="overlayBboxes">OD Boxes: -</div>
        <div id="overlayFps">Playback: -</div>
      </div>
      <div id="modeBar">
        <button class="modeButton" id="modeGt" type="button">GT</button>
        <button class="modeButton" id="modePred" type="button">PRED</button>
        <button class="modeButton" id="modeLoss" type="button">LOSSVIEW</button>
        <button class="modeButton" id="modeEdit" type="button">EDIT</button>
      </div>
      <div id="lossHistogramPanel">
        <div id="lossHistogramHeader">
          <div>Loss Histogram</div>
          <button id="lossHistogramClose" type="button">×</button>
        </div>
        <canvas id="lossHistogramCanvas" width="360" height="220"></canvas>
      </div>
      <div id="bboxPanel">
        <div id="bboxPanelHeader">
          <div id="bboxPanelTitle">Selected OD Box</div>
          <button id="bboxPanelClose" type="button">×</button>
        </div>
        <div id="bboxPanelBody"></div>
      </div>
      <div id="imageFloatingPanels"></div>
      <template id="imageFloatingPanelTemplate">
        <div class="imageFloatingPanel open">
          <div class="imageFloatingPanelHeader">
            <div class="imageFloatingPanelTitle">Frame Image</div>
            <div class="imageFloatingPanelControls">
              <select data-role="cameraSelect">
                <option value="">No image cameras</option>
              </select>
              <button data-role="close" type="button">×</button>
            </div>
          </div>
          <div class="imageFloatingPanelBody">
            <div class="imageFloatingPanelState">No image cameras available for this frame.</div>
            <img class="imageFloatingPanelImg is-hidden" alt="Frame camera view" />
            <div class="imageFloatingPanelResizeHandle"></div>
          </div>
        </div>
      </template>
      <div id="logPanel">
        <div id="logPanelHeader">
          <div>Viewer Log</div>
          <div id="logPanelControls">
            <button id="logPanelMinimize" type="button">−</button>
          </div>
        </div>
        <div id="logPanelBody">
          <div class="logSection" id="frameLogSection">
            <div class="logSectionTitle">Frame Log</div>
            <div id="frameLogContent"></div>
          </div>
          <div class="logSection" id="eventLogSection">
            <div class="logSectionTitle">Event Log</div>
            <div id="eventLogViewport">
              <div id="eventLogContent"></div>
            </div>
          </div>
        </div>
        <div class="logResizeHandle n" data-dir="n"></div>
        <div class="logResizeHandle s" data-dir="s"></div>
        <div class="logResizeHandle e" data-dir="e"></div>
        <div class="logResizeHandle w" data-dir="w"></div>
        <div class="logResizeHandle nw" data-dir="nw"></div>
        <div class="logResizeHandle ne" data-dir="ne"></div>
        <div class="logResizeHandle sw" data-dir="sw"></div>
        <div class="logResizeHandle se" data-dir="se"></div>
      </div>
    </main>
  </div>

  <script type="module">
    import * as THREE from "./static/three.module.js";
    import { OrbitControls } from "./static/OrbitControls.js";

    const decoder = new TextDecoder();
    const APP_BASE = new URL(".", window.location.href);
    function appUrl(path) {
      return new URL(path.replace(/^\\//, ""), APP_BASE).toString();
    }
    const frameCache = new Map();
    const detCache = new Map();
    const frameInfoCache = new Map();
    const flowCache = new Map();
    const staticCache = new Map();
    const lossCache = new Map();
    const ignoreCache = new Map();
    const subBoxFlowCache = new Map();
    const prevPointsCache = new Map();
    const frameImagesCache = new Map();
    const imageBlobCache = new Map();
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const state = {
      meta: null,
      index: 0,
      playing: false,
      fps: 5,
      pointSize: 2,
      autoFit: false,
      showDetections: true,
      showFutureBoxes: true,
      showFlow: false,
      showSubBoxFlow: false,
      showSubBoxFlowText: false,
      showStaticLabels: false,
      showPrevPoints: false,
      showLossView: false,
      editMode: false,
      hideLossBelowThreshold: false,
      hideIgnorePoints: false,
      detLineThickness: 0.12,
      detScoreThreshold: 0.0,
      lossRatioPercent: 75.0,
      lossThreshold: 0.0,
      currentMesh: null,
      currentFutureFlowMesh: null,
      currentPrevMesh: null,
      currentBoxes: null,
      currentSelectionBeam: null,
      currentSubBoxFlow: null,
      currentFrameData: null,
      currentStaticLabels: null,
      currentLossData: null,
      currentIgnoreMask: null,
      currentLossStats: null,
      currentLossRange: null,
      editRectangles: {},
      editDraftStart: null,
      currentEditRectangle: null,
      currentEditPreview: null,
      currentName: "-",
      currentPoints: 0,
      currentAboveThresholdCount: 0,
      currentBoxCount: 0,
      lossHistogramDismissed: false,
      forceRefitOnNextFrame: false,
      labelSource: "gt",
      logMinimized: false,
      imagePanelSources: [],
      imagePanelFrameIndex: -1,
      imageRefreshToken: 0,
      imagePanelDefaultCameraKey: "",
      imagePanelWindows: [],
      nextImagePanelWindowId: 1,
      selectedDetection: null,
      pointerDown: null,
      lastTickMs: performance.now(),
      loading: null,
      subdirs: [],
    };
    const VIEWER_UI_PREFERENCES_KEY = "mi-pyvista-vis-multi-browser-ui-preferences-v1";
    const DEFAULT_UI_PREFERENCES = Object.freeze({
      fps: 5,
      pointSize: 2,
      autoFit: false,
      showDetections: true,
      showFutureBoxes: true,
      showFlow: false,
      showSubBoxFlow: false,
      showSubBoxFlowText: false,
      showStaticLabels: false,
      hideLossBelowThreshold: false,
      hideIgnorePoints: false,
      detLineThickness: 0.12,
      detScoreThreshold: 0.0,
      lossRatioPercent: 75.0,
      lossThreshold: 0.0,
      sidebarWidth: 340,
      logMinimized: false,
      imagePanelWindows: [],
    });

    const elements = {
      rootDirInput: document.getElementById("rootDirInput"),
      scanRoot: document.getElementById("scanRoot"),
      subdirSelect: document.getElementById("subdirSelect"),
      loadSelected: document.getElementById("loadSelected"),
      frameSlider: document.getElementById("frameSlider"),
      frameValue: document.getElementById("frameValue"),
      frameJumpInput: document.getElementById("frameJumpInput"),
      jumpFrame: document.getElementById("jumpFrame"),
      framePercentInput: document.getElementById("framePercentInput"),
      jumpPercent: document.getElementById("jumpPercent"),
      jumpRandom: document.getElementById("jumpRandom"),
      fpsSlider: document.getElementById("fpsSlider"),
      fpsValue: document.getElementById("fpsValue"),
      pointSizeSlider: document.getElementById("pointSizeSlider"),
      pointSizeValue: document.getElementById("pointSizeValue"),
      playPause: document.getElementById("playPause"),
      prevFrame: document.getElementById("prevFrame"),
      nextFrame: document.getElementById("nextFrame"),
      togglePrevPoints: document.getElementById("togglePrevPoints"),
      resetView: document.getElementById("resetView"),
      toggleAutoFit: document.getElementById("toggleAutoFit"),
      toggleDetections: document.getElementById("toggleDetections"),
      toggleFutureBoxes: document.getElementById("toggleFutureBoxes"),
      toggleFlow: document.getElementById("toggleFlow"),
      toggleSubBoxFlow: document.getElementById("toggleSubBoxFlow"),
      toggleSubBoxFlowText: document.getElementById("toggleSubBoxFlowText"),
      toggleStaticLabels: document.getElementById("toggleStaticLabels"),
      clearEditCurrent: document.getElementById("clearEditCurrent"),
      saveEditRectangles: document.getElementById("saveEditRectangles"),
      editControls: document.getElementById("editControls"),
      editStatus: document.getElementById("editStatus"),
      imagePanelCameraSelect: document.getElementById("imagePanelCameraSelect"),
      addImagePanelWindow: document.getElementById("addImagePanelWindow"),
      openImagePanel: document.getElementById("openImagePanel"),
      imagePanelStatus: document.getElementById("imagePanelStatus"),
      toggleHideLossBelowThreshold: document.getElementById("toggleHideLossBelowThreshold"),
      toggleHideIgnorePoints: document.getElementById("toggleHideIgnorePoints"),
      lossControls: document.getElementById("lossControls"),
      lossRatioSlider: document.getElementById("lossRatioSlider"),
      lossRatioValue: document.getElementById("lossRatioValue"),
      lossRatioInput: document.getElementById("lossRatioInput"),
      lossThresholdSlider: document.getElementById("lossThresholdSlider"),
      lossThresholdValue: document.getElementById("lossThresholdValue"),
      lossThresholdInput: document.getElementById("lossThresholdInput"),
      lossRangeInfo: document.getElementById("lossRangeInfo"),
      lossAboveCount: document.getElementById("lossAboveCount"),
      lossHistogramPanel: document.getElementById("lossHistogramPanel"),
      lossHistogramClose: document.getElementById("lossHistogramClose"),
      lossHistogramCanvas: document.getElementById("lossHistogramCanvas"),
      detLineThicknessInput: document.getElementById("detLineThicknessInput"),
      detLineThicknessValue: document.getElementById("detLineThicknessValue"),
      detScoreSlider: document.getElementById("detScoreSlider"),
      detScoreValue: document.getElementById("detScoreValue"),
      detScoreInput: document.getElementById("detScoreInput"),
      status: document.getElementById("status"),
      legend: document.getElementById("legend"),
      overlayName: document.getElementById("overlayName"),
      overlayPoints: document.getElementById("overlayPoints"),
      overlayBboxes: document.getElementById("overlayBboxes"),
      overlayFps: document.getElementById("overlayFps"),
      modeGt: document.getElementById("modeGt"),
      modePred: document.getElementById("modePred"),
      modeLoss: document.getElementById("modeLoss"),
      modeEdit: document.getElementById("modeEdit"),
      bboxPanel: document.getElementById("bboxPanel"),
      bboxPanelBody: document.getElementById("bboxPanelBody"),
      bboxPanelClose: document.getElementById("bboxPanelClose"),
      imageFloatingPanels: document.getElementById("imageFloatingPanels"),
      imageFloatingPanelTemplate: document.getElementById("imageFloatingPanelTemplate"),
      sidebar: document.getElementById("sidebar"),
      sidebarHeader: document.getElementById("sidebarHeader"),
      sidebarSplitter: document.getElementById("sidebarSplitter"),
      logPanel: document.getElementById("logPanel"),
      logPanelHeader: document.getElementById("logPanelHeader"),
      logPanelBody: document.getElementById("logPanelBody"),
      logPanelMinimize: document.getElementById("logPanelMinimize"),
      frameLogContent: document.getElementById("frameLogContent"),
      eventLogViewport: document.getElementById("eventLogViewport"),
      eventLogContent: document.getElementById("eventLogContent"),
      pklPathEntries: document.getElementById("pklPathEntries"),
      evalDirEntries: document.getElementById("evalDirEntries"),
      canvasWrap: document.getElementById("canvasWrap"),
    };

    function clampStoredNumber(value, min, max, fallback) {
      if (!Number.isFinite(value)) {
        return fallback;
      }
      return Math.min(max, Math.max(min, value));
    }

    function getSidebarWidthPreference() {
      const raw = getComputedStyle(document.documentElement)
        .getPropertyValue("--sidebar-width")
        .trim();
      const parsed = Number.parseFloat(raw);
      return clampStoredNumber(
        parsed,
        280,
        560,
        DEFAULT_UI_PREFERENCES.sidebarWidth
      );
    }

    function createDefaultImagePanelRect(index = 0) {
      return {
        left: 24 + index * 28,
        top: 88 + index * 24,
        width: 480,
        height: 320,
      };
    }

    function normalizeImagePanelRect(rawRect, index = 0) {
      const fallbackRect = createDefaultImagePanelRect(index);
      return {
        left: clampStoredNumber(Number(rawRect?.left), 0, 2000, fallbackRect.left),
        top: clampStoredNumber(Number(rawRect?.top), 0, 2000, fallbackRect.top),
        width: clampStoredNumber(Number(rawRect?.width), 280, 1600, fallbackRect.width),
        height: clampStoredNumber(Number(rawRect?.height), 180, 1200, fallbackRect.height),
      };
    }

    function normalizeImagePanelWindow(source, index) {
      const windowSource =
        source && typeof source === "object" ? source : {};
      return {
        id: Math.max(
          1,
          Math.round(
            clampStoredNumber(Number(windowSource.id), 1, 100000, index + 1)
          )
        ),
        cameraKey:
          typeof windowSource.cameraKey === "string"
            ? windowSource.cameraKey
            : typeof windowSource.imagePanelCameraKey === "string"
            ? windowSource.imagePanelCameraKey
            : "",
        rect: normalizeImagePanelRect(
          windowSource.rect ?? windowSource.imagePanelRect,
          index
        ),
        objectUrl: null,
        loading: false,
        status: "No image cameras available for this frame.",
      };
    }

    function buildStoredImagePanelWindow(windowState, index) {
      const normalized = normalizeImagePanelWindow(windowState, index);
      return {
        id: normalized.id,
        cameraKey: normalized.cameraKey,
        rect: normalized.rect,
      };
    }

    function normalizeUiPreferences(rawPreferences) {
      const source =
        rawPreferences && typeof rawPreferences === "object" ? rawPreferences : {};
      const rawImagePanelWindows = Array.isArray(source.imagePanelWindows)
        ? source.imagePanelWindows
        : source.imagePanelOpen
        ? [
            {
              id: 1,
              cameraKey: source.imagePanelCameraKey,
              rect: source.imagePanelRect,
            },
          ]
        : [];
      const imagePanelWindows = [];
      const seenImagePanelWindowIds = new Set();
      rawImagePanelWindows.forEach((entry, index) => {
        let normalizedWindow = normalizeImagePanelWindow(entry, index);
        while (seenImagePanelWindowIds.has(normalizedWindow.id)) {
          normalizedWindow = {
            ...normalizedWindow,
            id: normalizedWindow.id + 1,
          };
        }
        seenImagePanelWindowIds.add(normalizedWindow.id);
        imagePanelWindows.push(normalizedWindow);
      });
      return {
        fps: Math.round(
          clampStoredNumber(
            Number(source.fps),
            1,
            20,
            DEFAULT_UI_PREFERENCES.fps
          )
        ),
        pointSize: clampStoredNumber(
          Number(source.pointSize),
          1,
          8,
          DEFAULT_UI_PREFERENCES.pointSize
        ),
        autoFit: Boolean(source.autoFit),
        showDetections: Boolean(source.showDetections),
        showFutureBoxes: source.showFutureBoxes !== false,
        showFlow: Boolean(source.showFlow),
        showSubBoxFlow: Boolean(source.showSubBoxFlow),
        showSubBoxFlowText: Boolean(source.showSubBoxFlowText),
        showStaticLabels: Boolean(source.showStaticLabels),
        hideLossBelowThreshold: Boolean(source.hideLossBelowThreshold),
        hideIgnorePoints: Boolean(source.hideIgnorePoints),
        detLineThickness: clampStoredNumber(
          Number(source.detLineThickness),
          0.02,
          1.0,
          DEFAULT_UI_PREFERENCES.detLineThickness
        ),
        detScoreThreshold: clampStoredNumber(
          Number(source.detScoreThreshold),
          0.0,
          1.0,
          DEFAULT_UI_PREFERENCES.detScoreThreshold
        ),
        lossRatioPercent: clampStoredNumber(
          Number(source.lossRatioPercent),
          0.0,
          100.0,
          DEFAULT_UI_PREFERENCES.lossRatioPercent
        ),
        lossThreshold: Number.isFinite(Number(source.lossThreshold))
          ? Number(source.lossThreshold)
          : DEFAULT_UI_PREFERENCES.lossThreshold,
        sidebarWidth: clampStoredNumber(
          Number(source.sidebarWidth),
          280,
          560,
          DEFAULT_UI_PREFERENCES.sidebarWidth
        ),
        logMinimized: Boolean(source.logMinimized),
        imagePanelWindows,
      };
    }

    function buildUiPreferences() {
      return normalizeUiPreferences({
        fps: state.fps,
        pointSize: state.pointSize,
        autoFit: state.autoFit,
        showDetections: state.showDetections,
        showFutureBoxes: state.showFutureBoxes,
        showFlow: state.showFlow,
        showSubBoxFlow: state.showSubBoxFlow,
        showSubBoxFlowText: state.showSubBoxFlowText,
        showStaticLabels: state.showStaticLabels,
        hideLossBelowThreshold: state.hideLossBelowThreshold,
        hideIgnorePoints: state.hideIgnorePoints,
        detLineThickness: state.detLineThickness,
        detScoreThreshold: state.detScoreThreshold,
        lossRatioPercent: state.lossRatioPercent,
        lossThreshold: state.lossThreshold,
        sidebarWidth: getSidebarWidthPreference(),
        logMinimized: state.logMinimized,
        imagePanelWindows: state.imagePanelWindows.map((windowState, index) =>
          buildStoredImagePanelWindow(windowState, index)
        ),
      });
    }

    function loadUiPreferences() {
      try {
        const raw = localStorage.getItem(VIEWER_UI_PREFERENCES_KEY);
        if (!raw) {
          return null;
        }
        return normalizeUiPreferences(JSON.parse(raw));
      } catch (error) {
        console.warn("Failed to load UI preferences.", error);
        return null;
      }
    }

    function saveUiPreferences() {
      try {
        localStorage.setItem(
          VIEWER_UI_PREFERENCES_KEY,
          JSON.stringify(buildUiPreferences())
        );
      } catch (error) {
        console.warn("Failed to save UI preferences.", error);
        appendLog(`UI preference save skipped: ${error.message}`, "warn");
      }
    }

    function applyStoredUiPreferences() {
      const storedPreferences = loadUiPreferences();
      if (!storedPreferences) {
        return;
      }
      state.fps = storedPreferences.fps;
      state.pointSize = storedPreferences.pointSize;
      state.autoFit = storedPreferences.autoFit;
      state.showDetections = storedPreferences.showDetections;
      state.showFutureBoxes = storedPreferences.showFutureBoxes;
      state.showFlow = storedPreferences.showFlow;
      state.showSubBoxFlow = storedPreferences.showSubBoxFlow;
      state.showSubBoxFlowText = storedPreferences.showSubBoxFlowText;
      state.showStaticLabels = storedPreferences.showStaticLabels;
      state.hideLossBelowThreshold = storedPreferences.hideLossBelowThreshold;
      state.hideIgnorePoints = storedPreferences.hideIgnorePoints;
      state.detLineThickness = storedPreferences.detLineThickness;
      state.detScoreThreshold = storedPreferences.detScoreThreshold;
      state.lossRatioPercent = storedPreferences.lossRatioPercent;
      state.lossThreshold = storedPreferences.lossThreshold;
      state.logMinimized = storedPreferences.logMinimized;
      state.imagePanelWindows = storedPreferences.imagePanelWindows.map((windowState, index) =>
        normalizeImagePanelWindow(windowState, index)
      );
      state.nextImagePanelWindowId =
        state.imagePanelWindows.reduce(
          (maxId, windowState) => Math.max(maxId, windowState.id),
          0
        ) + 1;
      document.documentElement.style.setProperty(
        "--sidebar-width",
        `${storedPreferences.sidebarWidth}px`
      );
      elements.fpsSlider.value = String(state.fps);
      elements.pointSizeSlider.value = String(state.pointSize);
      elements.detScoreSlider.value = String(state.detScoreThreshold);
      elements.detLineThicknessInput.value = state.detLineThickness.toFixed(2);
      elements.lossRatioSlider.value = String(state.lossRatioPercent);
      elements.lossRatioInput.value = state.lossRatioPercent.toFixed(4);
      elements.lossThresholdInput.value = state.lossThreshold.toFixed(4);
      elements.logPanel.classList.toggle("minimized", state.logMinimized);
      elements.logPanelMinimize.textContent = state.logMinimized ? "+" : "−";
      syncImagePanelSelectors();
      renderImagePanelWindows();
    }

    function revokeImagePanelWindowObjectUrl(windowState) {
      if (!windowState?.objectUrl) {
        return;
      }
      URL.revokeObjectURL(windowState.objectUrl);
      windowState.objectUrl = null;
    }

    function setImagePanelStateMessage(message) {
      elements.imagePanelStatus.value = message;
    }

    function getImagePanelSource(cameraKey) {
      if (!cameraKey) {
        return null;
      }
      return (
        state.imagePanelSources.find((source) => source.cameraKey === cameraKey) ||
        null
      );
    }

    function populateImagePanelSelector(select, selectedCameraKey) {
      select.innerHTML = "";
      const sources = state.imagePanelSources || [];
      if (!sources.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No image cameras";
        select.appendChild(option);
        select.value = "";
        return;
      }
      const normalizedSelection = selectedCameraKey || "";
      const hasSelectedCamera = sources.some(
        (source) => source.cameraKey === normalizedSelection
      );
      if (normalizedSelection && !hasSelectedCamera) {
        const missingOption = document.createElement("option");
        missingOption.value = normalizedSelection;
        missingOption.textContent = `${normalizedSelection} (missing)`;
        select.appendChild(missingOption);
      }
      for (const source of sources) {
        const option = document.createElement("option");
        option.value = source.cameraKey;
        option.textContent = source.cameraLabel;
        select.appendChild(option);
      }
      select.value = normalizedSelection || sources[0].cameraKey;
    }

    function syncImagePanelSelectors() {
      if (!state.imagePanelDefaultCameraKey && state.imagePanelSources.length) {
        state.imagePanelDefaultCameraKey = state.imagePanelSources[0].cameraKey;
      }
      populateImagePanelSelector(
        elements.imagePanelCameraSelect,
        state.imagePanelDefaultCameraKey
      );
    }

    function getViewerRect() {
      return document.getElementById("viewer").getBoundingClientRect();
    }

    function clampImagePanelRect(rect) {
      const viewerRect = getViewerRect();
      const minWidth = 280;
      const minHeight = 180;
      const width = Math.max(minWidth, Math.min(rect.width, Math.max(minWidth, viewerRect.width)));
      const height = Math.max(minHeight, Math.min(rect.height, Math.max(minHeight, viewerRect.height)));
      const maxLeft = Math.max(0, viewerRect.width - width);
      const maxTop = Math.max(0, viewerRect.height - height);
      return {
        left: Math.min(Math.max(0, rect.left), maxLeft),
        top: Math.min(Math.max(0, rect.top), maxTop),
        width,
        height,
      };
    }

    function applyImagePanelWindowLayout(panelElement, rect) {
      panelElement.style.left = `${rect.left}px`;
      panelElement.style.top = `${rect.top}px`;
      panelElement.style.width = `${rect.width}px`;
      panelElement.style.height = `${rect.height}px`;
    }

    function setImagePanelWindowRect(windowId, nextRect, persist = false, panelElement = null) {
      const windowState = state.imagePanelWindows.find((entry) => entry.id === windowId);
      if (!windowState) {
        return;
      }
      windowState.rect = clampImagePanelRect(nextRect);
      if (panelElement) {
        applyImagePanelWindowLayout(panelElement, windowState.rect);
      } else {
        renderImagePanelWindows();
      }
      if (persist) {
        saveUiPreferences();
      }
    }

    function getImagePanelWindowTitle(windowState) {
      const source = getImagePanelSource(windowState.cameraKey);
      if (source) {
        return `Frame Image · ${source.cameraLabel}`;
      }
      if (windowState.cameraKey) {
        return `Frame Image · ${windowState.cameraKey}`;
      }
      return `Frame Image ${windowState.id}`;
    }

    function bindImagePanelWindowDrag(headerElement, panelElement, windowState) {
      headerElement.addEventListener("pointerdown", (event) => {
        if (event.target.closest("button") || event.target.closest("select")) {
          return;
        }
        event.preventDefault();
        const startRect = { ...windowState.rect };
        const startX = event.clientX;
        const startY = event.clientY;

        function onMove(moveEvent) {
          const dx = moveEvent.clientX - startX;
          const dy = moveEvent.clientY - startY;
          setImagePanelWindowRect(
            windowState.id,
            {
              ...startRect,
              left: startRect.left + dx,
              top: startRect.top + dy,
            },
            false,
            panelElement
          );
        }

        function onUp() {
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
          saveUiPreferences();
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
      });
    }

    function bindImagePanelWindowResize(handleElement, panelElement, windowState) {
      handleElement.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const startRect = { ...windowState.rect };
        const startX = event.clientX;
        const startY = event.clientY;

        function onMove(moveEvent) {
          const dx = moveEvent.clientX - startX;
          const dy = moveEvent.clientY - startY;
          setImagePanelWindowRect(
            windowState.id,
            {
              ...startRect,
              width: startRect.width + dx,
              height: startRect.height + dy,
            },
            false,
            panelElement
          );
        }

        function onUp() {
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
          saveUiPreferences();
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
      });
    }

    function renderImagePanelWindows() {
      const fragment = document.createDocumentFragment();
      for (const windowState of state.imagePanelWindows) {
        const panelFragment = elements.imageFloatingPanelTemplate.content.cloneNode(true);
        const panelElement = panelFragment.querySelector(".imageFloatingPanel");
        const headerElement = panelFragment.querySelector(".imageFloatingPanelHeader");
        const titleElement = panelFragment.querySelector(".imageFloatingPanelTitle");
        const selectElement = panelFragment.querySelector('[data-role="cameraSelect"]');
        const closeElement = panelFragment.querySelector('[data-role="close"]');
        const bodyElement = panelFragment.querySelector(".imageFloatingPanelBody");
        const stateElement = panelFragment.querySelector(".imageFloatingPanelState");
        const imageElement = panelFragment.querySelector(".imageFloatingPanelImg");
        const resizeElement = panelFragment.querySelector(".imageFloatingPanelResizeHandle");
        const stop = (event) => {
          event.stopPropagation();
        };

        panelElement.dataset.windowId = String(windowState.id);
        applyImagePanelWindowLayout(panelElement, windowState.rect);
        titleElement.textContent = getImagePanelWindowTitle(windowState);
        populateImagePanelSelector(selectElement, windowState.cameraKey);

        const shouldShowImage = Boolean(windowState.objectUrl) && !windowState.loading;
        if (shouldShowImage) {
          imageElement.src = windowState.objectUrl;
          imageElement.classList.remove("is-hidden");
          stateElement.classList.add("is-hidden");
        } else {
          imageElement.removeAttribute("src");
          imageElement.classList.add("is-hidden");
          stateElement.textContent = windowState.status;
          stateElement.classList.remove("is-hidden");
        }

        selectElement.addEventListener("change", (event) => {
          windowState.cameraKey = event.target.value || "";
          if (windowState.cameraKey) {
            state.imagePanelDefaultCameraKey = windowState.cameraKey;
          }
          syncImagePanelSelectors();
          renderImagePanelWindows();
          scheduleImagePanelRefresh(state.index);
          saveUiPreferences();
        });
        closeElement.addEventListener("click", () => {
          closeImagePanelWindow(windowState.id);
        });

        bindImagePanelWindowDrag(headerElement, panelElement, windowState);
        bindImagePanelWindowResize(resizeElement, panelElement, windowState);

        panelElement.addEventListener("pointerdown", stop);
        panelElement.addEventListener("wheel", stop, { passive: true });
        bodyElement.addEventListener("pointerdown", stop);
        fragment.appendChild(panelFragment);
      }
      elements.imageFloatingPanels.replaceChildren(fragment);
    }

    function createImagePanelWindow(cameraKey = "") {
      const windowState = normalizeImagePanelWindow(
        {
          id: state.nextImagePanelWindowId,
          cameraKey,
          rect: createDefaultImagePanelRect(state.imagePanelWindows.length),
        },
        state.imagePanelWindows.length
      );
      state.nextImagePanelWindowId = Math.max(
        state.nextImagePanelWindowId,
        windowState.id + 1
      );
      return windowState;
    }

    function buildImageBlobCacheKey(index, cameraKey) {
      return `${index}:${cameraKey}`;
    }

    function isImageRefreshStale(index, refreshToken) {
      return index !== state.index || refreshToken !== state.imageRefreshToken;
    }

    function prepareImagePanelWindowsForRefresh(index) {
      for (const windowState of state.imagePanelWindows) {
        revokeImagePanelWindowObjectUrl(windowState);
        windowState.loading = true;
        windowState.status = `Loading frame ${index + 1} image...`;
      }
      renderImagePanelWindows();
    }

    async function refreshSingleImagePanelWindow(windowState, index, refreshToken) {
      if (!state.imagePanelSources.length) {
        revokeImagePanelWindowObjectUrl(windowState);
        windowState.loading = false;
        windowState.status = "No image cameras available for this frame.";
        return;
      }
      if (!windowState.cameraKey) {
        windowState.cameraKey =
          state.imagePanelDefaultCameraKey || state.imagePanelSources[0].cameraKey;
      }
      const selectedSource = getImagePanelSource(windowState.cameraKey);
      if (!selectedSource) {
        revokeImagePanelWindowObjectUrl(windowState);
        windowState.loading = false;
        windowState.status = `Camera ${windowState.cameraKey || "selection"} is not available for this frame.`;
        return;
      }

      try {
        revokeImagePanelWindowObjectUrl(windowState);
        windowState.loading = true;
        windowState.status = `Loading ${selectedSource.cameraLabel}...`;
        const imageBlob = await loadImagePanelImage(index, selectedSource.cameraKey);
        if (isImageRefreshStale(index, refreshToken)) {
          return;
        }
        if (!imageBlob) {
          windowState.status = `Camera ${selectedSource.cameraLabel} is not available for this frame.`;
          return;
        }
        windowState.objectUrl = URL.createObjectURL(imageBlob);
        windowState.status = `Showing ${selectedSource.cameraLabel} on frame ${index + 1}`;
      } catch (error) {
        revokeImagePanelWindowObjectUrl(windowState);
        windowState.status = `Image panel refresh failed: ${error.message}`;
        appendLog(`Image panel refresh failed: ${error.message}`, "error");
      } finally {
        windowState.loading = false;
      }
    }

    async function preloadImagePanelImagesForFrame(index, cameraKeys) {
      const uniqueCameraKeys = [...new Set(cameraKeys.filter(Boolean))];
      await Promise.allSettled(
        uniqueCameraKeys.map((cameraKey) => loadImagePanelImage(index, cameraKey))
      );
    }

    async function refreshImagePanelForFrame(index = state.index, refreshToken = state.imageRefreshToken) {
      if (!state.meta) {
        return;
      }
      try {
        const imageSources = await fetchFrameImages(index);
        if (isImageRefreshStale(index, refreshToken)) {
          return;
        }
        state.imagePanelSources = imageSources;
        state.imagePanelFrameIndex = index;
        if (!state.imagePanelDefaultCameraKey && state.imagePanelSources.length) {
          state.imagePanelDefaultCameraKey = state.imagePanelSources[0].cameraKey;
        }
        syncImagePanelSelectors();
        if (!state.imagePanelWindows.length) {
          renderImagePanelWindows();
          if (!state.imagePanelSources.length) {
            setImagePanelStateMessage("No image cameras available for this frame.");
          } else {
            setImagePanelStateMessage(
              `Ready: ${state.imagePanelSources.length} image camera(s) available.`
            );
          }
          const prefetchCameraKey =
            state.imagePanelDefaultCameraKey || state.imagePanelSources[0]?.cameraKey || "";
          if (prefetchCameraKey) {
            preloadImagePanelImagesForFrame(index, [prefetchCameraKey]).catch(() => {});
          }
          return;
        }

        renderImagePanelWindows();
        await Promise.all(
          state.imagePanelWindows.map((windowState) =>
            refreshSingleImagePanelWindow(windowState, index, refreshToken)
          )
        );
        if (isImageRefreshStale(index, refreshToken)) {
          return;
        }
        renderImagePanelWindows();

        if (!state.imagePanelSources.length) {
          setImagePanelStateMessage("No image cameras available for this frame.");
          return;
        }
        setImagePanelStateMessage(
          `Showing ${state.imagePanelWindows.length} image window(s) on frame ${index + 1}.`
        );
      } catch (error) {
        for (const windowState of state.imagePanelWindows) {
          revokeImagePanelWindowObjectUrl(windowState);
          windowState.loading = false;
          windowState.status = `Image panel refresh failed: ${error.message}`;
        }
        renderImagePanelWindows();
        setImagePanelStateMessage(`Image panel refresh failed: ${error.message}`);
        appendLog(`Image panel refresh failed: ${error.message}`, "error");
      }
    }

    function scheduleImagePanelRefresh(index = state.index) {
      const refreshToken = ++state.imageRefreshToken;
      if (state.imagePanelWindows.length) {
        prepareImagePanelWindowsForRefresh(index);
      }
      refreshImagePanelForFrame(index, refreshToken).catch((error) => {
        appendLog(`Image panel refresh failed: ${error.message}`, "error");
      });
    }

    function addImagePanelWindow(cameraKey = state.imagePanelDefaultCameraKey) {
      const nextCameraKey =
        cameraKey || state.imagePanelDefaultCameraKey || state.imagePanelSources[0]?.cameraKey || "";
      const windowState = createImagePanelWindow(nextCameraKey);
      state.imagePanelWindows.push(windowState);
      renderImagePanelWindows();
      scheduleImagePanelRefresh(state.index);
      saveUiPreferences();
    }

    function closeImagePanelWindow(windowId) {
      const windowIndex = state.imagePanelWindows.findIndex((entry) => entry.id === windowId);
      if (windowIndex < 0) {
        return;
      }
      const [removedWindow] = state.imagePanelWindows.splice(windowIndex, 1);
      revokeImagePanelWindowObjectUrl(removedWindow);
      renderImagePanelWindows();
      if (!state.imagePanelWindows.length) {
        if (!state.imagePanelSources.length) {
          setImagePanelStateMessage("No image cameras available for this frame.");
        } else {
          setImagePanelStateMessage(
            `Ready: ${state.imagePanelSources.length} image camera(s) available.`
          );
        }
      } else {
        setImagePanelStateMessage(
          `Showing ${state.imagePanelWindows.length} image window(s) on frame ${state.index + 1}.`
        );
      }
      saveUiPreferences();
    }

    function openImageFloatingPanel() {
      addImagePanelWindow();
    }

    function closeImageFloatingPanel() {
      const lastWindow = state.imagePanelWindows[state.imagePanelWindows.length - 1];
      if (lastWindow) {
        closeImagePanelWindow(lastWindow.id);
      }
    }

    function initImageFloatingPanel() {
      syncImagePanelSelectors();
      renderImagePanelWindows();
    }

    async function fetchFrameImages(index) {
      if (frameImagesCache.has(index)) {
        return frameImagesCache.get(index);
      }
      const response = await fetch(appUrl(`/api/frame_images/${index}`));
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `Frame image metadata request failed: ${response.status}`);
      }
      const items = Array.isArray(payload.items) ? payload.items : [];
      frameImagesCache.set(index, items);
      if (frameImagesCache.size > 12) {
        const firstKey = frameImagesCache.keys().next().value;
        frameImagesCache.delete(firstKey);
      }
      return items;
    }

    async function loadImagePanelImage(index, cameraKey) {
      const cacheKey = buildImageBlobCacheKey(index, cameraKey);
      if (imageBlobCache.has(cacheKey)) {
        return imageBlobCache.get(cacheKey);
      }
      const response = await fetch(
        appUrl(`/api/image/${index}?camera_key=${encodeURIComponent(cameraKey)}`)
      );
      if (response.status === 404) {
        return null;
      }
      if (!response.ok) {
        throw new Error(`Image request failed: ${response.status}`);
      }
      const imageBlob = await response.blob();
      imageBlobCache.set(cacheKey, imageBlob);
      if (imageBlobCache.size > 24) {
        const firstKey = imageBlobCache.keys().next().value;
        imageBlobCache.delete(firstKey);
      }
      return imageBlob;
    }

    function initSidebarPanels() {
      const panels = document.querySelectorAll("#sidebar .block[data-panel-title]");
      for (const panel of panels) {
        const toggle = panel.querySelector(".sidebarPanelToggle");
        if (!toggle) {
          continue;
        }
        const applyPanelCollapsedState = (collapsed) => {
          panel.classList.toggle("is-collapsed", collapsed);
          toggle.setAttribute("aria-expanded", String(!collapsed));
        };
        applyPanelCollapsedState(panel.classList.contains("is-collapsed"));
        toggle.addEventListener("click", () => {
          applyPanelCollapsedState(!panel.classList.contains("is-collapsed"));
        });
      }
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 5000);
    camera.position.set(40, 18, 40);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    elements.canvasWrap.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    controls.rotateSpeed = 1.1;
    controls.panSpeed = 1.1;
    controls.zoomSpeed = 1.15;
    controls.mouseButtons.MIDDLE = null;
    controls.target.set(30, 0, 0);
    controls.addEventListener("start", () => {
      if (state.autoFit) {
        state.autoFit = false;
        updateLabels();
      }
    });
    controls.addEventListener("change", () => {
      updateCameraClippingPlanes();
    });

    const ambient = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambient);

    function createRoiGrid() {
      const xMin = -24;
      const xMax = 24;
      const yMin = -80;
      const yMax = 200;
      const step = 10;
      const zPlane = 0.01;
      const vertices = [];

      function pushLine(ax, ay, az, bx, by, bz) {
        vertices.push(ax, ay, az, bx, by, bz);
      }

      for (let x = xMin; x <= xMax + 1e-6; x += step) {
        pushLine(x, zPlane, -yMin, x, zPlane, -yMax);
      }
      for (let y = yMin; y <= yMax + 1e-6; y += step) {
        pushLine(xMin, zPlane, -y, xMax, zPlane, -y);
      }
      if ((xMax - xMin) % step !== 0) {
        pushLine(xMax, zPlane, -yMin, xMax, zPlane, -yMax);
      }
      if ((yMax - yMin) % step !== 0) {
        pushLine(xMin, zPlane, -yMax, xMax, zPlane, -yMax);
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
      const material = new THREE.LineBasicMaterial({
        color: 0x248eff,
        transparent: true,
        opacity: 0.6,
      });
      return new THREE.LineSegments(geometry, material);
    }

    function createOriginAxes() {
      const group = new THREE.Group();

      function makeArrow(color, direction) {
        const arrow = new THREE.ArrowHelper(
          direction.clone().normalize(),
          new THREE.Vector3(0, 0, 0),
          1.0,
          color,
          0.25,
          0.18
        );
        arrow.line.material.linewidth = 3;
        arrow.line.scale.setScalar(1.5);
        return arrow;
      }

      group.add(makeArrow(0xff3b30, new THREE.Vector3(1, 0, 0)));
      group.add(makeArrow(0x39ff14, new THREE.Vector3(0, 0, -1)));
      group.add(makeArrow(0x3b82f6, new THREE.Vector3(0, 1, 0)));
      return group;
    }

    scene.add(createRoiGrid());
    scene.add(createOriginAxes());

    function resize() {
      const width = elements.canvasWrap.clientWidth;
      const height = elements.canvasWrap.clientHeight;
      renderer.setSize(width, height);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    }
    window.addEventListener("resize", resize);
    resize();

    function setInputValueIfIdle(element, value) {
      if (document.activeElement === element) {
        return;
      }
      element.value = value;
    }

    function getFramePercent(index, total) {
      if (total <= 1) {
        return 0;
      }
      return (index / (total - 1)) * 100;
    }

    function hasCurrentLossData() {
      return Boolean(
        state.currentLossData &&
        state.currentLossData.count > 0 &&
        state.currentLossRange
      );
    }

    function getCurrentFrameKey() {
      return state.currentName || "";
    }

    function getCurrentPointModeLabel() {
      if (state.editMode) {
        return "EDIT";
      }
      return state.showLossView ? "LOSSVIEW" : state.labelSource.toUpperCase();
    }

    function setViewMode(mode) {
      const leavingEditMode = state.editMode && mode !== "edit";
      if (leavingEditMode) {
        state.editMode = false;
        state.editDraftStart = null;
        renderEditPreviewRectangle(null);
        renderEditCurrentRectangle(null);
        controls.enableRotate = true;
        state.forceRefitOnNextFrame = true;
      }
      if (mode === "edit") {
        state.editMode = true;
        state.showLossView = false;
        controls.enableRotate = false;
        alignCameraToBev();
        renderStatusRows(buildStatusRows());
        renderEditCurrentRectangle(getCurrentFrameRectangle());
        updateLabels();
        return;
      }
      if (mode === "loss") {
        if (!hasCurrentLossData()) {
          appendLog("Loss mode ignored: pts_point_loss missing for current frame.", "warn");
          return;
        }
        state.showLossView = true;
        state.lossHistogramDismissed = false;
        renderStatusRows(buildStatusRows());
        applyCurrentPointVisuals();
        updateLabels();
        return;
      }
      if (mode === "pred") {
        if (!state.meta?.hasEvalResults) {
          appendLog("PRED mode ignored: no eval results configured.", "warn");
          return;
        }
        state.labelSource = "pred";
      } else {
        state.labelSource = "gt";
      }
      if (state.showLossView) {
        state.showLossView = false;
      }
      refreshCurrentFrameVisuals();
      updateLabels();
    }

    function rectangleFromCorners(startPoint, endPoint) {
      const x1 = Math.min(startPoint.x, endPoint.x);
      const x2 = Math.max(startPoint.x, endPoint.x);
      const y1 = Math.min(startPoint.y, endPoint.y);
      const y2 = Math.max(startPoint.y, endPoint.y);
      return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
      ];
    }

    function formatLossNumber(value) {
      return Number(value).toFixed(4);
    }

    function formatLossRange(range) {
      if (!range) {
        return "N/A";
      }
      return `${formatLossNumber(range.min)} .. ${formatLossNumber(range.max)}`;
    }

    function computeLossRange(lossData) {
      if (!lossData || !lossData.count) {
        return null;
      }
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < lossData.count; i += 1) {
        const value = lossData.values[i];
        if (!Number.isFinite(value)) {
          continue;
        }
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
      if (!Number.isFinite(min) || !Number.isFinite(max)) {
        return null;
      }
      return { min, max };
    }

    function computeLossStats(lossData, ignoreMask) {
      if (!lossData || !lossData.count) {
        return null;
      }
      const values = [];
      for (let i = 0; i < lossData.count; i += 1) {
        if (ignoreMask && ignoreMask.count === lossData.count && ignoreMask.labels[i] > 0) {
          continue;
        }
        const value = lossData.values[i];
        if (!Number.isFinite(value)) {
          continue;
        }
        values.push(value);
      }
      if (!values.length) {
        return null;
      }
      values.sort((a, b) => a - b);
      return {
        values,
        count: values.length,
        range: { min: values[0], max: values[values.length - 1] },
      };
    }

    function clampLossRatioPercent(value) {
      if (!Number.isFinite(value)) {
        return state.lossRatioPercent;
      }
      return Math.min(100.0, Math.max(0.0, value));
    }

    function getLossThresholdFromPercentile(lossStats, ratioPercent) {
      if (!lossStats || !lossStats.count) {
        return 0.0;
      }
      const index = Math.min(
        lossStats.count - 1,
        Math.max(0, Math.floor((clampLossRatioPercent(ratioPercent) / 100.0) * lossStats.count))
      );
      return lossStats.values[index];
    }

    function getLossPercentileFromThreshold(lossStats, threshold) {
      if (!lossStats || !lossStats.count) {
        return state.lossRatioPercent;
      }
      let index = lossStats.count - 1;
      for (let i = 0; i < lossStats.count; i += 1) {
        if (lossStats.values[i] >= threshold) {
          index = i;
          break;
        }
      }
      return clampLossRatioPercent((index / lossStats.count) * 100.0);
    }

    function computeLossCoverageCurve(lossStats) {
      if (!lossStats || !lossStats.count) {
        return {
          points: [{ ratio: 0, coverage: 0 }, { ratio: 1, coverage: 1 }],
          totalLossSum: 0,
        };
      }
      const descending = [...lossStats.values].sort((a, b) => b - a);
      const totalLossSum = descending.reduce((acc, value) => acc + value, 0);
      if (!(totalLossSum > 0)) {
        return {
          points: [{ ratio: 0, coverage: 0 }, ...descending.map((_, index) => ({
            ratio: (index + 1) / descending.length,
            coverage: 0,
          }))],
          totalLossSum: 0,
        };
      }

      let keptLossSum = 0;
      const curve = [{ ratio: 0, coverage: 0 }];
      for (let i = 0; i < descending.length; i += 1) {
        keptLossSum += descending[i];
        curve.push({
          ratio: (i + 1) / descending.length,
          coverage: keptLossSum / totalLossSum, // kept_loss_sum / total_loss_sum
        });
      }
      return { points: curve, totalLossSum };
    }

    function clampLossThreshold(value, range) {
      if (!range) {
        return state.lossThreshold;
      }
      if (!Number.isFinite(value)) {
        return state.lossThreshold;
      }
      return Math.min(range.max, Math.max(range.min, value));
    }

    function buildStatusRows() {
      const rows = [
        { key: "PKL File", value: state.meta?.pklFile || "-" },
        { key: "Eval Dir", value: state.meta?.evalDir || "-" },
        { key: "Sample Rate", value: state.meta ? `every ${state.meta.sampleRate} point(s)` : "-" },
        { key: "Label Source", value: state.labelSource },
        { key: "Point Mode", value: getCurrentPointModeLabel() },
      ];
      if (state.editMode) {
        rows.push({
          key: "Edit Frame",
          value: getCurrentFrameKey() || "-",
        });
        rows.push({
          key: "Has Rectangle",
          value: String(Boolean(getCurrentFrameRectangle())),
        });
        rows.push({
          key: "Edit Save Path",
          value: String(DEFAULT_EDIT_RECTANGLES_PATH),
        });
      }
      if (state.meta?.hasPointLoss) {
        rows.push({
          key: "Loss Range",
          value: hasCurrentLossData() ? formatLossRange(state.currentLossRange) : "N/A for current frame",
        });
        rows.push({
          key: "Loss Threshold",
          value: hasCurrentLossData()
            ? `${formatLossNumber(state.lossThreshold)} (${state.lossRatioPercent.toFixed(4)}% percentile)`
            : "N/A",
        });
        rows.push({
          key: "Above Threshold Count",
          value: hasCurrentLossData() ? String(state.currentAboveThresholdCount) : "N/A",
        });
        rows.push({
          key: "Hide Below Threshold",
          value: state.showLossView ? String(state.hideLossBelowThreshold) : "LOSS mode only",
        });
        rows.push({
          key: "Hide Ignore Points",
          value: state.showLossView ? String(state.hideIgnorePoints) : "LOSS mode only",
        });
      }
      return rows;
    }

    function renderLossHistogram() {
      const shouldShow =
        state.showLossView &&
        hasCurrentLossData() &&
        !state.lossHistogramDismissed &&
        state.currentLossStats &&
        state.currentLossStats.count > 0;
      elements.lossHistogramPanel.classList.toggle("open", shouldShow);
      if (!shouldShow) {
        return;
      }

      const canvas = elements.lossHistogramCanvas;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(240, Math.round(rect.width || canvas.width));
      const height = Math.max(180, Math.round(rect.height || canvas.height));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "rgba(2, 8, 23, 0.92)";
      ctx.fillRect(0, 0, width, height);

      const padding = { left: 42, right: 16, top: 16, bottom: 28 };
      const plotWidth = Math.max(1, width - padding.left - padding.right);
      const plotHeight = Math.max(1, height - padding.top - padding.bottom);
      const range = state.currentLossStats.range;
      const coverageCurve = computeLossCoverageCurve(state.currentLossStats);

      ctx.strokeStyle = "rgba(148, 163, 184, 0.24)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding.left, padding.top);
      ctx.lineTo(padding.left, padding.top + plotHeight);
      ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
      ctx.stroke();

      ctx.strokeStyle = "rgba(255, 140, 0, 0.92)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < coverageCurve.points.length; i += 1) {
        const point = coverageCurve.points[i];
        const px = padding.left + point.ratio * plotWidth;
        const py = padding.top + (1 - point.coverage) * plotHeight;
        if (i === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();

      const thresholdX = padding.left + (state.lossRatioPercent / 100.0) * plotWidth;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(thresholdX, padding.top);
      ctx.lineTo(thresholdX, padding.top + plotHeight);
      ctx.stroke();

      ctx.fillStyle = "#cbd5e1";
      ctx.font = '12px "IBM Plex Mono", monospace';
      ctx.fillText("0.0000%", padding.left, height - 8);
      const maxText = "100.0000%";
      ctx.fillText(maxText, width - padding.right - ctx.measureText(maxText).width, height - 8);
      ctx.fillText("1.0", 8, padding.top + 4);
      ctx.fillText("0.0", 8, padding.top + plotHeight);
      const markerText = `thr=${formatLossNumber(state.lossThreshold)} ratio=${state.lossRatioPercent.toFixed(4)}%`;
      ctx.fillText(markerText, padding.left, 14);
      const totalLossText = `total=${coverageCurve.totalLossSum.toFixed(4)}`;
      ctx.fillText(totalLossText, width - padding.right - ctx.measureText(totalLossText).width, 14);
      const rangeText = `loss ${formatLossNumber(range.min)} .. ${formatLossNumber(range.max)}`;
      ctx.fillText(rangeText, padding.left, height - 8 - 16);
    }

    function updateLabels() {
      const total = state.meta ? state.meta.frameCount : 0;
      const currentMode = getCurrentPointModeLabel();
      elements.frameValue.textContent = `${state.index + 1} / ${total}`;
      setInputValueIfIdle(elements.frameJumpInput, String(state.index + 1));
      setInputValueIfIdle(elements.framePercentInput, getFramePercent(state.index, total).toFixed(2));
      elements.fpsValue.textContent = `${state.fps}`;
      elements.pointSizeValue.textContent = state.pointSize.toFixed(1);
      const detScoreText = state.detScoreThreshold.toFixed(2);
      elements.detScoreValue.textContent = detScoreText;
      elements.detScoreInput.value = detScoreText;
      const detThicknessText = state.detLineThickness.toFixed(2);
      elements.detLineThicknessInput.value = detThicknessText;
      elements.detLineThicknessValue.textContent = detThicknessText;
      const hasLoss = hasCurrentLossData();
      const lossRatioText = `${state.lossRatioPercent.toFixed(4)}%`;
      const lossThresholdText = formatLossNumber(state.lossThreshold);
      elements.lossRatioSlider.value = String(state.lossRatioPercent);
      elements.lossRatioValue.textContent = lossRatioText;
      setInputValueIfIdle(elements.lossRatioInput, state.lossRatioPercent.toFixed(4));
      elements.lossThresholdValue.textContent = lossThresholdText;
      setInputValueIfIdle(elements.lossThresholdInput, lossThresholdText);
      elements.lossRangeInfo.value = hasLoss ? formatLossRange(state.currentLossRange) : "N/A";
      elements.lossControls.style.display = state.showLossView && hasLoss ? "" : "none";
      elements.editControls.style.display = state.editMode ? "" : "none";
      elements.toggleHideLossBelowThreshold.disabled = !state.showLossView || !hasLoss;
      elements.toggleHideLossBelowThreshold.textContent = `Hide Below Threshold: ${state.hideLossBelowThreshold ? "On" : "Off"}`;
      elements.toggleHideIgnorePoints.disabled = !state.showLossView || !hasLoss;
      elements.toggleHideIgnorePoints.textContent = `Hide Ignore Points: ${state.hideIgnorePoints ? "On" : "Off"}`;
      const currentRectangle = getCurrentFrameRectangle();
      if (state.editMode) {
        if (state.editDraftStart) {
          elements.editStatus.value = `Drawing ${getCurrentFrameKey()}: select opposite corner`;
        } else if (currentRectangle) {
          elements.editStatus.value = `${getCurrentFrameKey()}: rectangle saved in session`;
        } else {
          elements.editStatus.value = `${getCurrentFrameKey() || "-"}: no rectangle`;
        }
      } else {
        elements.editStatus.value = "EDIT mode off";
      }
      elements.lossAboveCount.value = hasLoss ? state.currentAboveThresholdCount.toLocaleString() : "0";
      if (hasLoss) {
        const sliderMin = state.currentLossRange.min;
        const sliderMax = state.currentLossRange.max <= sliderMin
          ? sliderMin + 1e-6
          : state.currentLossRange.max;
        elements.lossThresholdSlider.min = String(sliderMin);
        elements.lossThresholdSlider.max = String(sliderMax);
        elements.lossThresholdSlider.value = String(
          Math.min(sliderMax, Math.max(sliderMin, state.lossThreshold))
        );
      } else {
        elements.lossThresholdSlider.min = "0";
        elements.lossThresholdSlider.max = "1";
        elements.lossThresholdSlider.value = "0";
      }
      elements.overlayName.textContent = `Frame: ${state.currentName}`;
      const totalPointCount = state.currentFrameData ? state.currentFrameData.pointCount : 0;
      elements.overlayPoints.textContent = `Points: ${state.currentPoints.toLocaleString()} / ${totalPointCount.toLocaleString()}`;
      elements.overlayBboxes.textContent = `OD Boxes: ${state.currentBoxCount.toLocaleString()}`;
      elements.overlayFps.textContent = `Playback: ${state.playing ? `${state.fps} FPS` : "Paused"} | Mode: ${currentMode}`;
      elements.playPause.textContent = state.playing ? "Pause" : "Play";
      elements.togglePrevPoints.textContent = `Prev Pts: ${state.showPrevPoints ? "On" : "Off"}`;
      elements.toggleAutoFit.textContent = `Auto Fit: ${state.autoFit ? "On" : "Off"}`;
      elements.toggleDetections.textContent = `OD Boxes: ${state.showDetections ? "On" : "Off"}`;
      elements.toggleFutureBoxes.textContent = `Future Boxes: ${state.showFutureBoxes ? "On" : "Off"}`;
      elements.toggleFlow.textContent = `Flow: ${state.showFlow ? "On" : "Off"}`;
      elements.toggleSubBoxFlow.textContent = `Sub-box Flow: ${state.showSubBoxFlow ? "On" : "Off"}`;
      elements.toggleSubBoxFlowText.textContent = `Flow Text: ${state.showSubBoxFlowText ? "On" : "Off"}`;
      elements.toggleSubBoxFlowText.disabled = !state.showSubBoxFlow;
      elements.toggleStaticLabels.textContent = state.meta?.hasStaticLabels
        ? `Static Labels: ${state.showStaticLabels ? "On" : "Off"}`
        : "Static Labels: N/A";
      elements.toggleStaticLabels.disabled = !state.meta?.hasStaticLabels || state.showLossView;
      elements.modeGt.classList.toggle("active", currentMode === "GT");
      elements.modePred.classList.toggle("active", currentMode === "PRED");
      elements.modeLoss.classList.toggle("active", currentMode === "LOSSVIEW");
      elements.modeEdit.classList.toggle("active", currentMode === "EDIT");
      elements.modePred.disabled = !state.meta?.hasEvalResults;
      elements.modeLoss.disabled = !hasLoss;
      renderLossHistogram();
    }

    function appendLog(message, level = "info") {
      const stamp = new Date().toLocaleTimeString("zh-CN", { hour12: false });
      const entry = document.createElement("div");
      entry.className = `logEntry ${level}`;
      entry.textContent = `[${stamp}] ${message}`;
      const shouldStick =
        elements.eventLogViewport.scrollTop + elements.eventLogViewport.clientHeight >=
        elements.eventLogViewport.scrollHeight - 24;
      elements.eventLogContent.appendChild(entry);
      while (elements.eventLogContent.children.length > 120) {
        elements.eventLogContent.removeChild(elements.eventLogContent.firstChild);
      }
      if (shouldStick) {
        elements.eventLogViewport.scrollTop = elements.eventLogViewport.scrollHeight;
      }
    }

    function toggleLogPanel() {
      state.logMinimized = !state.logMinimized;
      elements.logPanel.classList.toggle("minimized", state.logMinimized);
      elements.logPanelMinimize.textContent = state.logMinimized ? "+" : "−";
      saveUiPreferences();
    }

    function renderFrameLog(frameInfo) {
      elements.frameLogContent.innerHTML = "";
      const entries = frameInfo?.entries || [];
      for (const entry of entries) {
        const row = document.createElement("div");
        row.className = "logKeyVal";
        const key = document.createElement("div");
        key.className = "logKey";
        key.textContent = entry.key;
        const val = document.createElement("div");
        val.className = "logVal";
        val.textContent = entry.value;
        row.appendChild(key);
        row.appendChild(val);
        elements.frameLogContent.appendChild(row);
      }
    }

    function renderStatusRows(rows) {
      elements.status.innerHTML = rows
        .map(
          ({ key, value }) =>
            `<div class="statusRow"><div class="statusKey">${key}</div><div class="statusVal">${value}</div></div>`
        )
        .join("");
    }

    function renderPathEntries(target, payload) {
      target.innerHTML = "";
      const entries = payload?.entries || [];
      if (!entries.length) {
        const option = document.createElement("option");
        option.textContent = "(No entries)";
        option.disabled = true;
        option.selected = true;
        target.appendChild(option);
        target.classList.add("empty");
        return;
      }
      target.classList.remove("empty");
      for (const item of entries) {
        const option = document.createElement("option");
        option.value = item.path;
        option.dataset.path = item.path;
        option.dataset.isDir = item.isDir ? "1" : "0";
        option.textContent = `${item.isDir ? "[DIR]" : "[FILE]"} ${item.name}`;
        target.appendChild(option);
      }
    }

    async function refreshPathEntries(value, kind, target) {
      try {
        const response = await fetch(
          `/api/path_entries?path=${encodeURIComponent(value || "")}&kind=${encodeURIComponent(kind)}`
        );
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Failed to list directory entries");
        }
        renderPathEntries(target, payload);
      } catch (error) {
        target.innerHTML = "";
        const option = document.createElement("option");
        option.textContent = `(Failed: ${error.message})`;
        option.disabled = true;
        option.selected = true;
        target.appendChild(option);
        target.classList.add("empty");
        appendLog(`Path browser failed (${kind}): ${error.message}`, "error");
      }
    }

    function syncPathBrowsers() {
      refreshPathEntries(elements.rootDirInput.value, "pkl_file", elements.pklPathEntries);
      refreshPathEntries(elements.subdirSelect.value, "eval_dir", elements.evalDirEntries);
    }

    function applyPathEntry(textarea, target, option, kind) {
      if (!option || option.disabled) {
        return;
      }
      textarea.value = option.dataset.path || option.value || "";
      refreshPathEntries(textarea.value, kind, target);
    }

    function clampFrameIndex(index) {
      if (!state.meta) {
        return 0;
      }
      return Math.min(Math.max(0, index), Math.max(state.meta.frameCount - 1, 0));
    }

    function jumpToFrameIndex(index) {
      if (!state.meta) {
        return;
      }
      const nextIndex = clampFrameIndex(index);
      state.lastTickMs = performance.now();
      showFrame(nextIndex, { force: true });
    }

    function jumpToFrameInput() {
      const value = Number(elements.frameJumpInput.value);
      if (!Number.isFinite(value)) {
        appendLog(`Invalid frame id: ${elements.frameJumpInput.value}`, "warn");
        updateLabels();
        return;
      }
      const normalized = value <= 0 ? 0 : Math.round(value) - 1;
      jumpToFrameIndex(normalized);
    }

    function jumpToPercentInput() {
      if (!state.meta) {
        return;
      }
      const value = Number(elements.framePercentInput.value);
      if (!Number.isFinite(value)) {
        appendLog(`Invalid frame percent: ${elements.framePercentInput.value}`, "warn");
        updateLabels();
        return;
      }
      const clamped = Math.min(100, Math.max(0, value));
      const total = Math.max(state.meta.frameCount - 1, 0);
      jumpToFrameIndex(Math.round((clamped / 100) * total));
    }

    function jumpToRandomFrame() {
      if (!state.meta || state.meta.frameCount <= 1) {
        return;
      }
      let nextIndex = Math.floor(Math.random() * state.meta.frameCount);
      if (state.meta.frameCount > 1 && nextIndex === state.index) {
        nextIndex = (nextIndex + 1) % state.meta.frameCount;
      }
      appendLog(`Random jump to frame ${nextIndex + 1}.`);
      jumpToFrameIndex(nextIndex);
    }

    function initResizableLogPanel() {
      const minWidth = 280;
      const minHeight = 120;
      const handles = elements.logPanel.querySelectorAll(".logResizeHandle");
      for (const handle of handles) {
        handle.addEventListener("pointerdown", (event) => {
          if (state.logMinimized) {
            return;
          }
          event.preventDefault();
          const dir = handle.dataset.dir;
          const startRect = elements.logPanel.getBoundingClientRect();
          const startX = event.clientX;
          const startY = event.clientY;
          const startLeft = startRect.left;
          const startTop = startRect.top;
          const startWidth = startRect.width;
          const startHeight = startRect.height;
          const parentRect = document.getElementById("viewer").getBoundingClientRect();

          function onMove(moveEvent) {
            const dx = moveEvent.clientX - startX;
            const dy = moveEvent.clientY - startY;
            let width = startWidth;
            let height = startHeight;
            let left = startLeft;
            let top = startTop;

            if (dir.includes("e")) {
              width = Math.max(minWidth, startWidth + dx);
            }
            if (dir.includes("s")) {
              height = Math.max(minHeight, startHeight + dy);
            }
            if (dir.includes("w")) {
              width = Math.max(minWidth, startWidth - dx);
              left = startLeft + (startWidth - width);
            }
            if (dir.includes("n")) {
              height = Math.max(minHeight, startHeight - dy);
              top = startTop + (startHeight - height);
            }

            const maxLeft = parentRect.right - width;
            const maxTop = parentRect.bottom - height;
            left = Math.min(Math.max(parentRect.left, left), maxLeft);
            top = Math.min(Math.max(parentRect.top, top), maxTop);

            elements.logPanel.style.left = `${left - parentRect.left}px`;
            elements.logPanel.style.top = `${top - parentRect.top}px`;
            elements.logPanel.style.bottom = "auto";
            elements.logPanel.style.width = `${width}px`;
            elements.logPanel.style.height = `${height}px`;
          }

          function onUp() {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
          }

          window.addEventListener("pointermove", onMove);
          window.addEventListener("pointerup", onUp);
        });
      }
    }

    function initDraggableLogPanel() {
      elements.logPanelHeader.addEventListener("pointerdown", (event) => {
        if (state.logMinimized) {
          return;
        }
        if (event.target.closest("button")) {
          return;
        }
        event.preventDefault();
        const startRect = elements.logPanel.getBoundingClientRect();
        const startX = event.clientX;
        const startY = event.clientY;
        const startLeft = startRect.left;
        const startTop = startRect.top;
        const parentRect = document.getElementById("viewer").getBoundingClientRect();

        function onMove(moveEvent) {
          const dx = moveEvent.clientX - startX;
          const dy = moveEvent.clientY - startY;
          const width = startRect.width;
          const height = startRect.height;
          const maxLeft = parentRect.right - width;
          const maxTop = parentRect.bottom - height;
          const left = Math.min(Math.max(parentRect.left, startLeft + dx), maxLeft);
          const top = Math.min(Math.max(parentRect.top, startTop + dy), maxTop);

          elements.logPanel.style.left = `${left - parentRect.left}px`;
          elements.logPanel.style.top = `${top - parentRect.top}px`;
          elements.logPanel.style.bottom = "auto";
        }

        function onUp() {
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
          saveUiPreferences();
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
      });
    }

    function initSidebarResizer() {
      const minWidth = 280;
      const maxWidth = 560;
      elements.sidebarSplitter.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        const appRect = document.getElementById("app").getBoundingClientRect();

        function onMove(moveEvent) {
          const width = Math.min(
            maxWidth,
            Math.max(minWidth, moveEvent.clientX - appRect.left)
          );
          document.documentElement.style.setProperty("--sidebar-width", `${width}px`);
          resize();
        }

        function onUp() {
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
          saveUiPreferences();
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
      });
    }

    function trapLogPanelEvents() {
      const stop = (event) => {
        event.stopPropagation();
      };
      elements.logPanel.addEventListener("pointerdown", stop);
      elements.logPanel.addEventListener("wheel", stop, { passive: true });
      elements.logPanelBody.addEventListener("pointerdown", stop);
      elements.eventLogViewport.addEventListener("wheel", stop, { passive: true });
      elements.eventLogViewport.addEventListener("pointerdown", stop);
      elements.frameLogContent.addEventListener("pointerdown", stop);
      elements.eventLogContent.addEventListener("pointerdown", stop);
      elements.lossHistogramPanel.addEventListener("pointerdown", stop);
      elements.lossHistogramPanel.addEventListener("wheel", stop, { passive: true });
      elements.imageFloatingPanels.addEventListener("pointerdown", stop);
      elements.imageFloatingPanels.addEventListener("wheel", stop, { passive: true });
    }

    function renderLegend(classes) {
      elements.legend.innerHTML = "";
      for (const entry of classes) {
        const item = document.createElement("div");
        item.className = "legendItem";
        const swatch = document.createElement("div");
        swatch.className = "swatch";
        swatch.style.background = `rgb(${entry.color.join(",")})`;
        const text = document.createElement("div");
        text.textContent = `${entry.id}: ${entry.name}`;
        item.appendChild(swatch);
        item.appendChild(text);
        elements.legend.appendChild(item);
      }
    }

    function renderSubdirs() {}

    function formatVec3(values) {
      return values.map((value) => Number(value).toFixed(3)).join(", ");
    }

    function disposeCurrentSelectionBeam() {
      if (!state.currentSelectionBeam) {
        return;
      }
      for (const child of state.currentSelectionBeam.children) {
        if (child.geometry) {
          child.geometry.dispose();
        }
        if (child.material) {
          child.material.dispose();
        }
      }
      scene.remove(state.currentSelectionBeam);
      state.currentSelectionBeam = null;
    }

    function disposeCurrentFutureFlowMesh() {
      if (!state.currentFutureFlowMesh) {
        return;
      }
      state.currentFutureFlowMesh.geometry.dispose();
      state.currentFutureFlowMesh.material.dispose();
      scene.remove(state.currentFutureFlowMesh);
      state.currentFutureFlowMesh = null;
    }

    function disposeCurrentSubBoxFlow() {
      if (!state.currentSubBoxFlow) {
        return;
      }
      state.currentSubBoxFlow.traverse((child) => {
        if (child.line?.geometry) {
          child.line.geometry.dispose();
        }
        if (child.line?.material) {
          child.line.material.dispose();
        }
        if (child.cone?.geometry) {
          child.cone.geometry.dispose();
        }
        if (child.cone?.material) {
          child.cone.material.dispose();
        }
        if (child.geometry) {
          child.geometry.dispose();
        }
        if (child.material?.map) {
          child.material.map.dispose();
        }
        if (child.material) {
          child.material.dispose();
        }
      });
      for (const child of state.currentSubBoxFlow.children) {
        state.currentSubBoxFlow.remove(child);
      }
      scene.remove(state.currentSubBoxFlow);
      state.currentSubBoxFlow = null;
    }

    function createSubBoxFlowLabelSprite(text) {
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      canvas.width = 128;
      canvas.height = 48;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = "rgba(2, 8, 23, 0.78)";
      context.fillRect(0, 8, canvas.width, 32);
      context.font = '20px "IBM Plex Mono", monospace';
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillStyle = "#f8fafc";
      context.fillText(text, canvas.width / 2, canvas.height / 2);

      const texture = new THREE.CanvasTexture(canvas);
      texture.needsUpdate = true;
      const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
      });
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(0.9, 0.34, 1.0);
      return sprite;
    }

    function createSelectionBeam(det) {
      disposeCurrentSelectionBeam();

      const [x, y, z] = det.center;
      const [dx, dy, dz] = det.size;
      const topY = z + dz / 2;
      const beamHeight = 30.0;
      const color = new THREE.Color(0xff3b30);

      const group = new THREE.Group();

      const beamGeometry = new THREE.BoxGeometry(dx, beamHeight, dy);
      const beamMaterial = new THREE.ShaderMaterial({
        uniforms: {
          baseColor: { value: color },
          baseOpacity: { value: 0.32 },
        },
        vertexShader: `
          varying float vHeightT;

          void main() {
            vHeightT = position.y / ${beamHeight.toFixed(6)} + 0.5;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          uniform vec3 baseColor;
          uniform float baseOpacity;
          varying float vHeightT;

          void main() {
            float fade = 1.0 - smoothstep(0.55, 1.0, clamp(vHeightT, 0.0, 1.0));
            float alpha = baseOpacity * fade;
            if (alpha <= 0.001) {
              discard;
            }
            gl_FragColor = vec4(baseColor, alpha);
          }
        `,
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const beam = new THREE.Mesh(beamGeometry, beamMaterial);
      beam.position.set(x, topY + beamHeight / 2, -y);
      beam.rotation.y = -det.yaw;
      group.add(beam);

      state.currentSelectionBeam = group;
      scene.add(group);
    }

    function setSelectedDetection(entry) {
      state.selectedDetection = entry;
      if (!state.currentBoxes) {
        disposeCurrentSelectionBeam();
        elements.bboxPanel.classList.remove("open");
        elements.bboxPanelBody.innerHTML = "";
        return;
      }

      for (const child of state.currentBoxes.children) {
        if (child.userData.kind !== "bboxEdge") {
          continue;
        }
        const isSelected = entry && child.userData.detIndex === entry.detIndex;
        child.material.color.copy(child.userData.baseColor);
        child.material.opacity = isSelected ? 1.0 : 0.95;
      }

      if (!entry) {
        disposeCurrentSelectionBeam();
        elements.bboxPanel.classList.remove("open");
        elements.bboxPanelBody.innerHTML = "";
        return;
      }

      const det = entry.detection;
      createSelectionBeam(det);
      elements.bboxPanelBody.innerHTML = [
        ["Class", `${det.name} (id=${det.bboxClass})`],
        ["Center xyz", formatVec3(det.center)],
        ["Size dx dy dz", formatVec3(det.size)],
        ["Score", Number(det.score).toFixed(4)],
        ["Yaw", Number(det.yaw).toFixed(4)],
      ]
        .map(
          ([key, value]) =>
            `<div class="bboxRow"><div class="bboxKey">${key}</div><div class="bboxVal">${value}</div></div>`
        )
        .join("");
      elements.bboxPanel.classList.add("open");
    }

    function getCurrentFrameRectangle() {
      const frameKey = getCurrentFrameKey();
      return frameKey ? state.editRectangles[frameKey] || null : null;
    }

    function disposeObject3d(object) {
      if (!object) {
        return;
      }
      if (object.geometry) {
        object.geometry.dispose();
      }
      if (object.material) {
        object.material.dispose();
      }
      scene.remove(object);
    }

    function rectangleToSceneVertices(rectangle, height = 0.06) {
      const vertices = [];
      for (const point of rectangle) {
        vertices.push(point[0], height, -point[1]);
      }
      return vertices;
    }

    function createEditRectangleLine(rectangle, color, opacity = 1.0) {
      const vertices = rectangleToSceneVertices(rectangle);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute([...vertices, ...vertices.slice(0, 3)], 3));
      const material = new THREE.LineBasicMaterial({
        color,
        transparent: opacity < 1.0,
        opacity,
      });
      return new THREE.Line(geometry, material);
    }

    function renderEditCurrentRectangle(rectangle) {
      disposeObject3d(state.currentEditRectangle);
      state.currentEditRectangle = null;
      if (!state.editMode || !rectangle) {
        return;
      }
      state.currentEditRectangle = createEditRectangleLine(rectangle, 0x22d3ee, 1.0);
      scene.add(state.currentEditRectangle);
    }

    function renderEditPreviewRectangle(rectangle) {
      disposeObject3d(state.currentEditPreview);
      state.currentEditPreview = null;
      if (!state.editMode || !rectangle) {
        return;
      }
      state.currentEditPreview = createEditRectangleLine(rectangle, 0xf59e0b, 0.75);
      scene.add(state.currentEditPreview);
    }

    function getGroundPointFromPointerEvent(event) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
      const hitPoint = new THREE.Vector3();
      if (!raycaster.ray.intersectPlane(plane, hitPoint)) {
        return null;
      }
      return { x: hitPoint.x, y: -hitPoint.z };
    }

    function buildRenderedPointCloud(frame, staticLabels, ignoreMask) {
      const useLossView =
        state.showLossView &&
        hasCurrentLossData() &&
        state.currentLossData.count === frame.pointCount;
      const keepAllVisible = !useLossView || !state.hideLossBelowThreshold;
      const palette = [
        [255, 32, 16],
        [255, 255, 255],
      ];
      const ignoreColor = [128, 128, 128];
      const lowColor = [255, 255, 255];
      const highColor = [255, 140, 0];

      const keptPositions = [];
      const keptColors = [];
      let aboveThresholdCount = 0;
      let ignoredPointCount = 0;

      for (let i = 0; i < frame.pointCount; i += 1) {
        const isIgnored = Boolean(ignoreMask && ignoreMask.count === frame.pointCount && ignoreMask.labels[i] > 0);
        if (isIgnored) {
          ignoredPointCount += 1;
        }

        const isAboveThreshold =
          useLossView &&
          !isIgnored &&
          state.currentLossData.values[i] >= state.lossThreshold;
        if (isAboveThreshold) {
          aboveThresholdCount += 1;
        }

        if (useLossView && isIgnored && state.hideIgnorePoints) {
          continue;
        }
        if (useLossView && !isIgnored && !isAboveThreshold && !keepAllVisible) {
          continue;
        }

        const ps = i * 3;
        keptPositions.push(
          frame.positions[ps],
          frame.positions[ps + 2],
          -frame.positions[ps + 1]
        );

        let color = null;
        if (isIgnored) {
          color = ignoreColor;
        } else if (useLossView) {
          color = isAboveThreshold ? highColor : lowColor;
        } else if (state.showStaticLabels && staticLabels && staticLabels.count === frame.pointCount) {
          color = palette[Math.min(1, staticLabels.labels[i])];
        } else {
          color = [
            frame.colors[ps],
            frame.colors[ps + 1],
            frame.colors[ps + 2],
          ];
        }
        keptColors.push(color[0], color[1], color[2]);
      }

      return {
        positions: new Float32Array(keptPositions),
        colors: new Uint8Array(keptColors),
        shownCount: keptPositions.length / 3,
        totalCount: frame.pointCount,
        aboveThresholdCount,
        ignoredPointCount,
      };
    }

    function createSubBoxFlowOverlay(items) {
      disposeCurrentSubBoxFlow();
      if (!state.showSubBoxFlow || !Array.isArray(items) || !items.length) {
        return;
      }

      const futureDtSeconds = 0.5;
      const arrowHeadLength = 0.12;
      const arrowHeadWidth = 0.06;
      const group = new THREE.Group();
      for (const item of items) {
        const center = item.center || [];
        const flow = item.flow || [];
        if (center.length < 3 || flow.length < 3) {
          continue;
        }
        const start = new THREE.Vector3(center[0], center[2] + (item.heightOffset || 0.0), -center[1]);
        const direction = new THREE.Vector3(flow[0], flow[2], -flow[1]);
        const speedValue = direction.length();
        const arrowLength = speedValue * futureDtSeconds;
        if (!(speedValue > 0) || !(arrowLength > 0)) {
          continue;
        }
        const arrow = new THREE.ArrowHelper(
          direction.clone().normalize(),
          start,
          arrowLength,
          0x22d3ee,
          arrowHeadLength,
          arrowHeadWidth,
        );
        arrow.line.material.transparent = true;
        arrow.line.material.opacity = 0.92;
        arrow.line.material.depthWrite = false;
        arrow.cone.material.transparent = true;
        arrow.cone.material.opacity = 0.92;
        arrow.cone.material.depthWrite = false;
        group.add(arrow);

        if (state.showSubBoxFlowText) {
          const label = createSubBoxFlowLabelSprite(speedValue.toFixed(2));
          const labelPosition = start.clone().add(direction.clone().normalize().multiplyScalar(arrowLength + 0.18));
          labelPosition.y += 0.08;
          label.position.copy(labelPosition);
          group.add(label);
        }
      }

      if (!group.children.length) {
        return;
      }
      state.currentSubBoxFlow = group;
      scene.add(group);
    }

    function parseFrameBuffer(buffer) {
      const view = new DataView(buffer);
      const magic = decoder.decode(buffer.slice(0, 4));
      if (magic !== "PCD0") {
        throw new Error(`Bad frame magic: ${magic}`);
      }
      const pointCount = view.getUint32(4, true);
      const nameLen = view.getUint32(8, true);
      let offset = 16;
      const name = decoder.decode(buffer.slice(offset, offset + nameLen));
      offset += nameLen;
      offset += (4 - (nameLen % 4)) % 4;
      const positions = new Float32Array(buffer, offset, pointCount * 3);
      offset += pointCount * 3 * 4;
      const colors = new Uint8Array(buffer, offset, pointCount * 3);
      return { name, pointCount, positions, colors };
    }

    function parseVectorBuffer(buffer, magicName) {
      const view = new DataView(buffer);
      const magic = decoder.decode(buffer.slice(0, 4));
      if (magic !== magicName) {
        throw new Error(`Bad vector magic: ${magic}`);
      }
      const count = view.getUint32(4, true);
      const values = new Float32Array(buffer, 16, count * 3);
      return { count, values };
    }

    function parseStaticBuffer(buffer) {
      const view = new DataView(buffer);
      const magic = decoder.decode(buffer.slice(0, 4));
      if (magic !== "STA0") {
        throw new Error(`Bad static magic: ${magic}`);
      }
      const count = view.getUint32(4, true);
      const labels = new Uint8Array(buffer, 16, count);
      return { count, labels };
    }

    function parseIgnoreBuffer(buffer) {
      const view = new DataView(buffer);
      const magic = decoder.decode(buffer.slice(0, 4));
      if (magic !== "IGN0") {
        throw new Error(`Bad ignore magic: ${magic}`);
      }
      const count = view.getUint32(4, true);
      const labels = new Uint8Array(buffer, 16, count);
      return { count, labels };
    }

    function parseScalarBuffer(buffer, magicName) {
      const view = new DataView(buffer);
      const magic = decoder.decode(buffer.slice(0, 4));
      if (magic !== magicName) {
        throw new Error(`Bad scalar magic: ${magic}`);
      }
      const count = view.getUint32(4, true);
      const values = new Float32Array(buffer, 16, count);
      return { count, values };
    }

    function buildCacheKey(index, labelSource) {
      return `${labelSource}:${index}`;
    }

    async function fetchFrame(index, labelSource = state.labelSource) {
      const cacheKey = buildCacheKey(index, labelSource);
      if (frameCache.has(cacheKey)) {
        return frameCache.get(cacheKey);
      }
      const response = await fetch(appUrl(`api/frame/${index}?labelSource=${encodeURIComponent(labelSource)}`));
      if (!response.ok) {
        throw new Error(`Frame request failed: ${response.status}`);
      }
      const parsed = parseFrameBuffer(await response.arrayBuffer());
      frameCache.set(cacheKey, parsed);
      if (frameCache.size > 6) {
        const firstKey = frameCache.keys().next().value;
        frameCache.delete(firstKey);
      }
      return parsed;
    }

    async function fetchDetections(index, labelSource = state.labelSource) {
      const cacheKey = buildCacheKey(index, labelSource);
      if (detCache.has(cacheKey)) {
        return detCache.get(cacheKey);
      }
      const response = await fetch(appUrl(`api/det/${index}?labelSource=${encodeURIComponent(labelSource)}`));
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `Detection request failed: ${response.status}`);
      }
      const detections = payload.detections || [];
      detCache.set(cacheKey, detections);
      if (detCache.size > 6) {
        const firstKey = detCache.keys().next().value;
        detCache.delete(firstKey);
      }
      return detections;
    }

    async function fetchFrameInfo(index, labelSource = state.labelSource) {
      const cacheKey = buildCacheKey(index, labelSource);
      if (frameInfoCache.has(cacheKey)) {
        return frameInfoCache.get(cacheKey);
      }
      const response = await fetch(appUrl(`api/frame_info/${index}?labelSource=${encodeURIComponent(labelSource)}`));
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `Frame info request failed: ${response.status}`);
      }
      frameInfoCache.set(cacheKey, payload);
      if (frameInfoCache.size > 6) {
        const firstKey = frameInfoCache.keys().next().value;
        frameInfoCache.delete(firstKey);
      }
      return payload;
    }

    async function fetchFlow(index, labelSource = state.labelSource) {
      const cacheKey = buildCacheKey(index, labelSource);
      if (flowCache.has(cacheKey)) {
        return flowCache.get(cacheKey);
      }
      const response = await fetch(
        appUrl(`api/flow/${index}?labelSource=${encodeURIComponent(labelSource)}`)
      );
      if (response.status === 404) {
        return null;
      }
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || `Flow request failed: ${response.status}`);
      }
      const parsed = parseVectorBuffer(await response.arrayBuffer(), "FLW0");
      flowCache.set(cacheKey, parsed);
      if (flowCache.size > 6) {
        const firstKey = flowCache.keys().next().value;
        flowCache.delete(firstKey);
      }
      return parsed;
    }

    async function fetchSubBoxFlows(index, labelSource = state.labelSource) {
      const cacheKey = buildCacheKey(index, labelSource);
      if (subBoxFlowCache.has(cacheKey)) {
        return subBoxFlowCache.get(cacheKey);
      }
      const response = await fetch(
        appUrl(`api/sub_box_flow/${index}?labelSource=${encodeURIComponent(labelSource)}`)
      );
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `Sub-box flow request failed: ${response.status}`);
      }
      const items = payload.items || [];
      subBoxFlowCache.set(cacheKey, items);
      if (subBoxFlowCache.size > 6) {
        const firstKey = subBoxFlowCache.keys().next().value;
        subBoxFlowCache.delete(firstKey);
      }
      return items;
    }

    async function fetchStaticLabels(index, labelSource = state.labelSource) {
      const cacheKey = buildCacheKey(index, labelSource);
      if (staticCache.has(cacheKey)) {
        return staticCache.get(cacheKey);
      }
      const response = await fetch(
        appUrl(`api/static/${index}?labelSource=${encodeURIComponent(labelSource)}`)
      );
      if (response.status === 404) {
        return null;
      }
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || `Static request failed: ${response.status}`);
      }
      const parsed = parseStaticBuffer(await response.arrayBuffer());
      staticCache.set(cacheKey, parsed);
      if (staticCache.size > 6) {
        const firstKey = staticCache.keys().next().value;
        staticCache.delete(firstKey);
      }
      return parsed;
    }

    async function fetchPointLoss(index) {
      if (lossCache.has(index)) {
        return lossCache.get(index);
      }
      const response = await fetch(appUrl(`api/loss/${index}`));
      if (response.status === 404) {
        return null;
      }
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || `Point loss request failed: ${response.status}`);
      }
      const parsed = parseScalarBuffer(await response.arrayBuffer(), "LOS0");
      lossCache.set(index, parsed);
      if (lossCache.size > 6) {
        const firstKey = lossCache.keys().next().value;
        lossCache.delete(firstKey);
      }
      return parsed;
    }

    async function fetchIgnoreMask(index) {
      if (ignoreCache.has(index)) {
        return ignoreCache.get(index);
      }
      const response = await fetch(appUrl(`api/ignore/${index}`));
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || `Ignore mask request failed: ${response.status}`);
      }
      const parsed = parseIgnoreBuffer(await response.arrayBuffer());
      ignoreCache.set(index, parsed);
      if (ignoreCache.size > 6) {
        const firstKey = ignoreCache.keys().next().value;
        ignoreCache.delete(firstKey);
      }
      return parsed;
    }

    async function fetchPrevPoints(index) {
      if (prevPointsCache.has(index)) {
        return prevPointsCache.get(index);
      }
      const response = await fetch(appUrl(`api/prev_points/${index}`));
      if (response.status === 404) {
        return null;
      }
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || `Previous points request failed: ${response.status}`);
      }
      const parsed = parseFrameBuffer(await response.arrayBuffer());
      prevPointsCache.set(index, parsed);
      if (prevPointsCache.size > 6) {
        const firstKey = prevPointsCache.keys().next().value;
        prevPointsCache.delete(firstKey);
      }
      return parsed;
    }

    async function loadSelectedDir() {
      const pklFile = elements.rootDirInput.value.trim();
      const evalDir = elements.subdirSelect.value.trim();
      if (!pklFile) {
        renderStatusRows([{ key: "Status", value: "Please enter a PKL file first." }]);
        appendLog("Load skipped: missing PKL file path.", "warn");
        return;
      }

      elements.scanRoot.disabled = true;
      elements.loadSelected.disabled = true;
      appendLog(
        `Opening dataset: ${pklFile}${evalDir ? ` | eval: ${evalDir}` : ""}`
      );
      try {
        const response = await fetch(appUrl("api/open"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pklFile, evalDir }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Failed to open PKL dataset");
        }

        frameCache.clear();
        detCache.clear();
        frameInfoCache.clear();
        flowCache.clear();
        staticCache.clear();
        lossCache.clear();
        ignoreCache.clear();
        subBoxFlowCache.clear();
        prevPointsCache.clear();
        frameImagesCache.clear();
        imageBlobCache.clear();
        disposeCurrentMesh();
        disposeCurrentFutureFlowMesh();
        disposeCurrentPrevMesh();
        disposeCurrentBoxes();
        disposeCurrentSubBoxFlow();
        setSelectedDetection(null);
        state.meta = payload.meta;
        state.index = 0;
        state.playing = false;
        state.labelSource = state.meta.labelSource || "gt";
        state.showStaticLabels = Boolean(state.meta.hasStaticLabels);
        state.showSubBoxFlow = false;
        state.showPrevPoints = false;
        state.showLossView = false;
        state.editMode = false;
        controls.enableRotate = true;
        state.hideLossBelowThreshold = false;
        state.hideIgnorePoints = false;
        state.lossHistogramDismissed = false;
        state.editRectangles = {};
        state.editDraftStart = null;
        state.currentName = "-";
        state.currentPoints = 0;
        state.currentAboveThresholdCount = 0;
        state.currentBoxCount = 0;
        state.currentStaticLabels = null;
        state.currentSubBoxFlow = null;
        state.currentPrevMesh = null;
        state.currentLossData = null;
        state.currentIgnoreMask = null;
        state.currentLossStats = null;
        state.currentLossRange = null;
        state.imageRefreshToken += 1;
        disposeObject3d(state.currentEditRectangle);
        state.currentEditRectangle = null;
        disposeObject3d(state.currentEditPreview);
        state.currentEditPreview = null;
        elements.rootDirInput.value = payload.meta.pklFile || "";
        elements.subdirSelect.value = payload.meta.evalDir || "";
        syncPathBrowsers();
        elements.frameSlider.max = String(Math.max(state.meta.frameCount - 1, 0));
        state.fps = Math.round(state.meta.fps);
        state.pointSize = Number(state.meta.pointSize);
        state.showStaticLabels = Boolean(state.meta.hasStaticLabels);
        applyStoredUiPreferences();
        state.lastTickMs = performance.now();
        renderLegend(state.meta.classes);
        appendLog(`Dataset loaded: ${state.meta.pklFile}`);
        appendLog(`Mode ${state.labelSource.toUpperCase()}${state.meta.evalDir ? `, eval dir: ${state.meta.evalDir}` : ""}`);
        await showFrame(0, { force: true });
      } catch (error) {
        renderStatusRows([{ key: "Error", value: `Failed to load PKL dataset: ${error.message}` }]);
        appendLog(`Dataset load failed: ${error.message}`, "error");
      } finally {
        elements.scanRoot.disabled = false;
        elements.loadSelected.disabled = false;
      }
    }

    function disposeCurrentMesh() {
      if (!state.currentMesh) {
        return;
      }
      state.currentMesh.geometry.dispose();
      state.currentMesh.material.dispose();
      scene.remove(state.currentMesh);
      state.currentMesh = null;
    }

    function disposeCurrentPrevMesh() {
      if (!state.currentPrevMesh) {
        return;
      }
      state.currentPrevMesh.geometry.dispose();
      state.currentPrevMesh.material.dispose();
      scene.remove(state.currentPrevMesh);
      state.currentPrevMesh = null;
    }

    function disposeCurrentBoxes() {
      disposeCurrentSelectionBeam();
      if (!state.currentBoxes) {
        return;
      }
      for (const child of state.currentBoxes.children) {
        if (child.geometry) {
          child.geometry.dispose();
        }
        if (child.material) {
          child.material.dispose();
        }
      }
      scene.remove(state.currentBoxes);
      state.currentBoxes = null;
    }

    function makePointsMaterial() {
      return new THREE.PointsMaterial({
        size: state.pointSize,
        vertexColors: true,
        sizeAttenuation: false,
        transparent: false,
        opacity: 1.0,
        toneMapped: false,
      });
    }

    function makePrevPointsMaterial() {
      return new THREE.PointsMaterial({
        size: Math.max(1, state.pointSize),
        vertexColors: true,
        sizeAttenuation: false,
        transparent: true,
        opacity: 0.38,
        depthWrite: false,
        toneMapped: false,
      });
    }

    function createFutureFlowPointOverlay(frame, flowData) {
      disposeCurrentFutureFlowMesh();
      if (!state.showFlow || !frame || !flowData || flowData.count !== frame.pointCount) {
        return;
      }

      const futureDtSeconds = 0.5;
      const futurePositions = [];
      const futureColors = [];
      for (let i = 0; i < frame.pointCount; i += 1) {
        const offset = i * 3;
        const flowX = flowData.values[offset];
        const flowY = flowData.values[offset + 1];
        const flowZ = flowData.values[offset + 2];
        if (flowX === 0.0 && flowY === 0.0 && flowZ === 0.0) {
          continue;
        }
        futurePositions.push(
          frame.positions[offset] + flowX * futureDtSeconds,
          frame.positions[offset + 2] + flowZ * futureDtSeconds,
          -(frame.positions[offset + 1] + flowY * futureDtSeconds)
        );
        futureColors.push(200, 80, 0);
      }

      if (!futurePositions.length) {
        return;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(futurePositions), 3)
      );
      geometry.setAttribute(
        "color",
        new THREE.Uint8BufferAttribute(new Uint8Array(futureColors), 3, true)
      );
      state.currentFutureFlowMesh = new THREE.Points(geometry, makePrevPointsMaterial());
      scene.add(state.currentFutureFlowMesh);
    }

    function renderPrevPoints(frame) {
      disposeCurrentPrevMesh();
      if (!state.showPrevPoints || !frame || frame.pointCount <= 0) {
        return;
      }
      const positions = new Float32Array(frame.pointCount * 3);
      for (let i = 0; i < frame.pointCount; i += 1) {
        const src = i * 3;
        positions[src] = frame.positions[src];
        positions[src + 1] = frame.positions[src + 2];
        positions[src + 2] = -frame.positions[src + 1];
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.Uint8BufferAttribute(frame.colors, 3, true));
      state.currentPrevMesh = new THREE.Points(geometry, makePrevPointsMaterial());
      scene.add(state.currentPrevMesh);
    }

    function applyCurrentPointVisuals({ fit = false } = {}) {
      if (!state.currentFrameData) {
        return;
      }
      const pointCloud = buildRenderedPointCloud(
        state.currentFrameData,
        state.currentStaticLabels,
        state.currentIgnoreMask
      );
      state.currentPoints = pointCloud.shownCount;
      state.currentAboveThresholdCount = pointCloud.aboveThresholdCount;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(pointCloud.positions, 3));
      geometry.setAttribute("color", new THREE.Uint8BufferAttribute(pointCloud.colors, 3, true));

      const points = new THREE.Points(geometry, makePointsMaterial());
      disposeCurrentMesh();
      state.currentMesh = points;
      scene.add(points);

      if (pointCloud.shownCount > 0) {
        if (fit) {
          fitCameraFromGeometry(geometry);
        } else {
          updateCameraClippingPlanes(geometry);
        }
      }
    }

    function getFilteredDetections(detections) {
      return detections.filter((det) => {
        if (Number(det.score) < state.detScoreThreshold) {
          return false;
        }
        if (!state.showFutureBoxes && det.isNextFrame) {
          return false;
        }
        return true;
      });
    }

    function clampDetScoreThreshold(value) {
      if (!Number.isFinite(value)) {
        return state.detScoreThreshold;
      }
      return Math.min(1.0, Math.max(0.0, value));
    }

    function setDetScoreThreshold(value) {
      state.detScoreThreshold = clampDetScoreThreshold(value);
      elements.detScoreSlider.value = String(state.detScoreThreshold);
      refreshCurrentDetections();
      updateLabels();
      saveUiPreferences();
    }

    function clampDetLineThickness(value) {
      if (!Number.isFinite(value)) {
        return state.detLineThickness;
      }
      return Math.min(1.0, Math.max(0.02, value));
    }

    function setDetLineThickness(value) {
      state.detLineThickness = clampDetLineThickness(value);
      refreshCurrentDetections();
      updateLabels();
      saveUiPreferences();
    }

    function createEdgeCylinder(start, end, radius, color, opacity = 0.95) {
      const startVec = new THREE.Vector3(...start);
      const endVec = new THREE.Vector3(...end);
      const delta = new THREE.Vector3().subVectors(endVec, startVec);
      const length = delta.length();
      const geometry = new THREE.CylinderGeometry(radius, radius, Math.max(length, 1e-4), 10);
      const material = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(startVec).add(endVec).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        delta.clone().normalize()
      );
      return mesh;
    }

    function createBoundingBoxLines(detections) {
      const visibleDetections = getFilteredDetections(detections);
      disposeCurrentBoxes();
      setSelectedDetection(null);
      state.currentBoxCount = visibleDetections.length;
      if (!visibleDetections.length) {
        return;
      }

      const edgePairs = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
      ];
      const faceTriangles = [
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
      ];
      const localCorners = [
        [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5],
      ];
      const group = new THREE.Group();

      visibleDetections.forEach((det, detIndex) => {
        const [cx, cy, cz] = det.center;
        const [dx, dy, dz] = det.size;
        const cosYaw = Math.cos(det.yaw);
        const sinYaw = Math.sin(det.yaw);
        const detCorners = localCorners.map(([sx, sy, sz]) => {
          const lx = sx * dx;
          const ly = sy * dy;
          const lz = sz * dz;
          const ox = cx + lx * cosYaw - ly * sinYaw;
          const oy = cy + lx * sinYaw + ly * cosYaw;
          const oz = cz + lz;
          return [ox, oz, -oy];
        });

        const linePositions = [];
        for (const [startIdx, endIdx] of edgePairs) {
          linePositions.push(...detCorners[startIdx], ...detCorners[endIdx]);
        }

        const edgeColor = det.isNextFrame
          ? new THREE.Color(1.0, 0.33, 0.1)
          : new THREE.Color(...det.color.map((value) => value / 255));
        const edgeOpacity = det.isNextFrame ? 0.65 : 0.95;
        for (let edgeIdx = 0; edgeIdx < linePositions.length; edgeIdx += 6) {
          const edge = createEdgeCylinder(
            linePositions.slice(edgeIdx, edgeIdx + 3),
            linePositions.slice(edgeIdx + 3, edgeIdx + 6),
            state.detLineThickness / 2,
            edgeColor,
            edgeOpacity
          );
          edge.userData = {
            kind: "bboxEdge",
            detIndex,
            detection: det,
            baseColor: edgeColor.clone(),
          };
          group.add(edge);
        }

        const pickPositions = [];
        for (const [a, b, c] of faceTriangles) {
          pickPositions.push(...detCorners[a], ...detCorners[b], ...detCorners[c]);
        }
        const pickGeometry = new THREE.BufferGeometry();
        pickGeometry.setAttribute("position", new THREE.Float32BufferAttribute(pickPositions, 3));
        pickGeometry.computeVertexNormals();
        const pickMaterial = new THREE.MeshBasicMaterial({
          transparent: true,
          opacity: 0.0,
          side: THREE.DoubleSide,
          depthWrite: false,
        });
        const pickMesh = new THREE.Mesh(pickGeometry, pickMaterial);
        pickMesh.userData = {
          kind: "bboxPick",
          detIndex,
          detection: det,
        };
        group.add(pickMesh);
      });

      state.currentBoxes = group;
      state.currentBoxes.visible = state.showDetections;
      scene.add(state.currentBoxes);
    }

    function handleCanvasClick(event) {
      if (state.editMode) {
        const groundPoint = getGroundPointFromPointerEvent(event);
        if (!groundPoint) {
          return;
        }
        if (!state.editDraftStart) {
          state.editDraftStart = groundPoint;
          renderEditPreviewRectangle(rectangleFromCorners(groundPoint, groundPoint));
          elements.editStatus.value = `Start corner set for ${getCurrentFrameKey() || "-"}`;
          return;
        }

        const rectangle = rectangleFromCorners(state.editDraftStart, groundPoint);
        const width = Math.abs(rectangle[1][0] - rectangle[0][0]);
        const height = Math.abs(rectangle[3][1] - rectangle[0][1]);
        if (width < 1e-6 || height < 1e-6) {
          appendLog("Edit rectangle ignored: width or height is zero.", "warn");
          state.editDraftStart = null;
          renderEditPreviewRectangle(null);
          renderEditCurrentRectangle(getCurrentFrameRectangle());
          updateLabels();
          return;
        }
        state.editRectangles[getCurrentFrameKey()] = rectangle;
        state.editDraftStart = null;
        renderEditPreviewRectangle(null);
        renderEditCurrentRectangle(rectangle);
        appendLog(`Rectangle updated for ${getCurrentFrameKey()}.`);
        updateLabels();
        return;
      }
      if (!state.showDetections || !state.currentBoxes) {
        return;
      }
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);

      const pickMeshes = state.currentBoxes.children.filter(
        (child) => child.userData.kind === "bboxPick"
      );
      const intersects = raycaster.intersectObjects(pickMeshes, false);
      if (!intersects.length) {
        setSelectedDetection(null);
        return;
      }
      const target = intersects[0].object.userData;
      setSelectedDetection({
        detIndex: target.detIndex,
        detection: target.detection,
      });
    }

    function focusViewOnPointer(event) {
      if (event.button !== 1) {
        return false;
      }
      if (!state.currentMesh) {
        return false;
      }
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      raycaster.params.Points.threshold = Math.max(1.0, state.pointSize * 0.75);

      let intersects = raycaster.intersectObject(state.currentMesh, false);
      if (!intersects.length && state.currentPrevMesh) {
        intersects = raycaster.intersectObject(state.currentPrevMesh, false);
      }
      if (!intersects.length) {
        return false;
      }

      const hitPoint = intersects[0].point.clone();
      const cameraOffset = camera.position.clone().sub(controls.target);
      controls.target.copy(hitPoint);
      camera.position.copy(hitPoint).add(cameraOffset);
      controls.update();
      updateCameraClippingPlanes();
      return true;
    }

    function updateCameraClippingPlanes(geometry = null) {
      const targetGeometry = geometry || state.currentMesh?.geometry;
      if (!targetGeometry) {
        return;
      }
      if (!targetGeometry.boundingBox) {
        targetGeometry.computeBoundingBox();
      }
      const box = targetGeometry.boundingBox;
      if (!box) {
        return;
      }

      const sphere = new THREE.Sphere();
      box.getBoundingSphere(sphere);
      const radius = Math.max(sphere.radius, 8);
      const distanceToCenter = camera.position.distanceTo(sphere.center);
      camera.near = Math.max(0.1, distanceToCenter / 1000);
      camera.far = Math.max(camera.near + 1, distanceToCenter + radius * 8);
      camera.updateProjectionMatrix();
    }

    function fitCameraFromGeometry(geometry) {
      geometry.computeBoundingBox();
      const box = geometry.boundingBox;
      if (!box) {
        return;
      }

      const center = new THREE.Vector3();
      box.getCenter(center);
      const sphere = new THREE.Sphere();
      box.getBoundingSphere(sphere);

      const vFov = THREE.MathUtils.degToRad(camera.fov);
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
      const limitingFov = Math.min(vFov, hFov);
      const radius = Math.max(sphere.radius, 8);
      const distance = (radius / Math.sin(limitingFov / 2)) * 0.58;
      const pitchFromNegativeY = THREE.MathUtils.degToRad(30);
      const offsetUp = Math.sin(pitchFromNegativeY) * distance;
      const offsetBack = Math.cos(pitchFromNegativeY) * distance;

      controls.target.copy(center);
      camera.up.set(0, 1, 0);
      camera.position.set(
        center.x,
        center.y + offsetUp,
        center.z + offsetBack
      );
      updateCameraClippingPlanes(geometry);
      camera.lookAt(center);
      controls.update();
      updateCameraClippingPlanes(geometry);
    }

    function alignCameraToBev() {
      const geometry = state.currentMesh?.geometry;
      if (!geometry) {
        return;
      }
      geometry.computeBoundingBox();
      const box = geometry.boundingBox;
      if (!box) {
        return;
      }
      const center = new THREE.Vector3();
      box.getCenter(center);
      const sphere = new THREE.Sphere();
      box.getBoundingSphere(sphere);
      const distance = Math.max(24, sphere.radius * 1.8);
      controls.target.copy(center);
      camera.up.set(0, 0, -1);
      camera.position.set(center.x, center.y + distance, center.z);
      camera.lookAt(center);
      controls.update();
      updateCameraClippingPlanes(geometry);
    }

    async function showFrame(index, { force = false } = {}) {
      if (!state.meta) {
        return;
      }
      const normalized = ((index % state.meta.frameCount) + state.meta.frameCount) % state.meta.frameCount;
      if (!force && normalized === state.index && state.currentMesh) {
        return;
      }
      if (state.loading) {
        return;
      }

      state.editDraftStart = null;
      renderEditPreviewRectangle(null);
      disposeCurrentFutureFlowMesh();

      const promises = [
        fetchFrame(normalized, state.labelSource),
        fetchDetections(normalized, state.labelSource).catch((error) => {
          appendLog(
            `Detection fetch failed for frame ${normalized + 1}: ${error.message}`,
            "error"
          );
          return [];
        }),
        fetchFrameInfo(normalized, state.labelSource).catch((error) => {
          appendLog(
            `Frame info fetch failed for frame ${normalized + 1}: ${error.message}`,
            "error"
          );
          return {
            entries: [
              { key: "FRAME", value: `${normalized + 1}` },
              { key: "MODE", value: state.labelSource.toUpperCase() },
              { key: "FRAME_INFO_ERROR", value: error.message },
            ],
          };
        }),
      ];
      if (state.meta?.hasStaticLabels) {
        promises.push(
          fetchStaticLabels(normalized, state.labelSource).catch((error) => {
            appendLog(
              `Static fetch failed for frame ${normalized + 1}: ${error.message}`,
              "error"
            );
            return null;
          })
        );
      } else {
        promises.push(Promise.resolve(null));
      }
      if (state.showFlow) {
        promises.push(
          fetchFlow(normalized, state.labelSource).catch((error) => {
            appendLog(
              `Flow fetch failed for frame ${normalized + 1}: ${error.message}`,
              "error"
            );
            return null;
          })
        );
      } else {
        promises.push(Promise.resolve(null));
      }
      if (state.showSubBoxFlow) {
        promises.push(
          fetchSubBoxFlows(normalized, state.labelSource).catch((error) => {
            appendLog(
              `Sub-box flow fetch failed for frame ${normalized + 1}: ${error.message}`,
              "error"
            );
            return [];
          })
        );
      } else {
        promises.push(Promise.resolve([]));
      }
      if (state.meta?.hasPointLoss) {
        promises.push(
          fetchPointLoss(normalized).catch((error) => {
            appendLog(
              `Point loss fetch failed for frame ${normalized + 1}: ${error.message}`,
              "error"
            );
            return null;
          })
        );
      } else {
        promises.push(Promise.resolve(null));
      }
      promises.push(
        fetchIgnoreMask(normalized).catch((error) => {
          appendLog(
            `Ignore mask fetch failed for frame ${normalized + 1}: ${error.message}`,
            "error"
          );
          return null;
        })
      );
      if (state.showPrevPoints) {
        promises.push(
          fetchPrevPoints(normalized).catch((error) => {
            appendLog(
              `Previous points fetch failed for frame ${normalized + 1}: ${error.message}`,
              "error"
            );
            return null;
          })
        );
      } else {
        promises.push(Promise.resolve(null));
      }
      state.loading = Promise.all(promises);
      try {
        const [frame, detections, frameInfo, staticLabels, flowData, subBoxFlows, pointLossData, ignoreMask, prevPoints] = await state.loading;
        state.index = normalized;
        state.currentName = frame.name;
        state.currentFrameData = frame;
        state.currentStaticLabels = staticLabels;
        state.currentLossData = pointLossData;
        state.currentIgnoreMask = ignoreMask;
        state.currentLossStats =
          pointLossData && pointLossData.count === frame.pointCount
            ? computeLossStats(pointLossData, ignoreMask)
            : null;
        state.currentLossRange = state.currentLossStats ? state.currentLossStats.range : null;
        if (state.currentLossStats) {
          state.lossThreshold = getLossThresholdFromPercentile(state.currentLossStats, state.lossRatioPercent);
        } else if (state.showLossView) {
          state.showLossView = false;
          appendLog(`Loss view disabled for frame ${state.index + 1}: pts_point_loss missing.`, "warn");
        }
        renderFrameLog(frameInfo);
        const shouldFit = !state.currentMesh || state.autoFit || state.forceRefitOnNextFrame;
        renderPrevPoints(prevPoints);
        applyCurrentPointVisuals({ fit: shouldFit });
        createFutureFlowPointOverlay(frame, flowData);
        createBoundingBoxLines(detections);
        createSubBoxFlowOverlay(subBoxFlows);
        if (state.editMode) {
          alignCameraToBev();
          renderEditCurrentRectangle(getCurrentFrameRectangle());
        }
        state.forceRefitOnNextFrame = false;

        elements.frameSlider.value = String(state.index);
        scheduleImagePanelRefresh(state.index);
        renderStatusRows(buildStatusRows());
        updateLabels();
        if (force || !state.playing) {
          appendLog(`Frame ${state.index + 1}/${state.meta.frameCount} loaded in ${getCurrentPointModeLabel()} mode.`);
        }

        const nextIndex = (state.index + 1) % state.meta.frameCount;
        fetchFrame(nextIndex, state.labelSource).catch(() => {});
        fetchDetections(nextIndex, state.labelSource).catch(() => {});
        fetchFrameInfo(nextIndex, state.labelSource).catch(() => {});
        if (state.meta?.hasStaticLabels) {
          fetchStaticLabels(nextIndex, state.labelSource).catch(() => {});
        }
        if (state.showFlow) {
          fetchFlow(nextIndex, state.labelSource).catch(() => {});
        }
        if (state.showSubBoxFlow) {
          fetchSubBoxFlows(nextIndex, state.labelSource).catch(() => {});
        }
        if (state.meta?.hasPointLoss) {
          fetchPointLoss(nextIndex).catch(() => {});
        }
        fetchIgnoreMask(nextIndex).catch(() => {});
        fetchFrameImages(nextIndex).catch(() => {});
      } catch (error) {
        renderStatusRows([{ key: "Error", value: `Failed to load frame: ${error.message}` }]);
        appendLog(`Frame load failed: ${error.message}`, "error");
        throw error;
      } finally {
        state.loading = null;
      }
    }

    async function refreshCurrentDetections() {
      if (!state.meta || state.loading) {
        return;
      }
      try {
        const detections = await fetchDetections(state.index, state.labelSource);
        createBoundingBoxLines(detections);
        updateLabels();
      } catch (error) {
        console.error(error);
        appendLog(`Detection refresh failed: ${error.message}`, "error");
      }
    }

    async function refreshCurrentFrameVisuals() {
      if (!state.meta || state.loading) {
        return;
      }
      try {
        await showFrame(state.index, { force: true });
      } catch (error) {
        appendLog(`Frame refresh failed: ${error.message}`, "error");
      }
    }

    async function init() {
      const response = await fetch(appUrl("api/meta"));
      state.meta = await response.json();
      state.labelSource = state.meta.labelSource || "gt";
      state.fps = Math.round(state.meta.fps);
      state.pointSize = Number(state.meta.pointSize);
      state.showLossView = false;
      state.editMode = false;
      controls.enableRotate = true;
      state.hideLossBelowThreshold = false;
      state.hideIgnorePoints = false;
      state.lossHistogramDismissed = false;
      state.editRectangles = {};
      state.editDraftStart = null;
      state.currentStaticLabels = null;
      state.showSubBoxFlow = false;
      state.currentSubBoxFlow = null;
      state.showPrevPoints = false;
      state.currentLossData = null;
      state.currentIgnoreMask = null;
      state.currentLossStats = null;
      state.currentLossRange = null;
      state.currentPrevMesh = null;
      state.currentEditRectangle = null;
      state.currentEditPreview = null;
      elements.rootDirInput.value = state.meta.pklFile || "";
      elements.subdirSelect.value = state.meta.evalDir || "";
      syncPathBrowsers();

      elements.frameSlider.max = String(Math.max(state.meta.frameCount - 1, 0));
      state.showStaticLabels = Boolean(state.meta.hasStaticLabels);
      applyStoredUiPreferences();
      renderLegend(state.meta.classes);
      initSidebarPanels();
      initResizableLogPanel();
      initDraggableLogPanel();
      initImageFloatingPanel();
      initSidebarResizer();
      trapLogPanelEvents();
      syncPathBrowsers();
      renderStatusRows(buildStatusRows());
      appendLog(`Viewer ready. Initial mode: ${state.labelSource.toUpperCase()}.`);
      await showFrame(0, { force: true });
    }

    elements.playPause.addEventListener("click", () => {
      if (!state.playing && state.meta && state.index >= state.meta.frameCount - 1) {
        showFrame(0, { force: true });
      }
      state.playing = !state.playing;
      state.lastTickMs = performance.now();
      updateLabels();
    });
    elements.prevFrame.addEventListener("click", () => showFrame(state.index - 1, { force: true }));
    elements.nextFrame.addEventListener("click", () => showFrame(state.index + 1, { force: true }));
    elements.togglePrevPoints.addEventListener("click", async () => {
      if (!state.meta) {
        return;
      }
      state.showPrevPoints = !state.showPrevPoints;
      if (!state.showPrevPoints) {
        disposeCurrentPrevMesh();
        updateLabels();
        return;
      }
      try {
        const prevPoints = await fetchPrevPoints(state.index);
        renderPrevPoints(prevPoints);
      } catch (error) {
        appendLog(`Previous points toggle failed: ${error.message}`, "error");
        state.showPrevPoints = false;
        disposeCurrentPrevMesh();
      }
      updateLabels();
    });
    elements.resetView.addEventListener("click", () => {
      if (state.currentMesh) {
        if (state.editMode) {
          alignCameraToBev();
        } else {
          fitCameraFromGeometry(state.currentMesh.geometry);
        }
      }
    });
    elements.toggleAutoFit.addEventListener("click", () => {
      state.autoFit = !state.autoFit;
      updateLabels();
      saveUiPreferences();
    });
    elements.toggleDetections.addEventListener("click", () => {
      state.showDetections = !state.showDetections;
      if (state.currentBoxes) {
        state.currentBoxes.visible = state.showDetections;
      }
      if (!state.showDetections) {
        setSelectedDetection(null);
      }
      updateLabels();
      saveUiPreferences();
    });
    elements.toggleFutureBoxes.addEventListener("click", () => {
      state.showFutureBoxes = !state.showFutureBoxes;
      refreshCurrentDetections();
      updateLabels();
      saveUiPreferences();
    });
    elements.toggleFlow.addEventListener("click", () => {
      state.showFlow = !state.showFlow;
      refreshCurrentFrameVisuals();
      updateLabels();
      saveUiPreferences();
    });
    elements.toggleSubBoxFlow.addEventListener("click", () => {
      state.showSubBoxFlow = !state.showSubBoxFlow;
      if (!state.showSubBoxFlow) {
        state.showSubBoxFlowText = false;
      }
      refreshCurrentFrameVisuals();
      updateLabels();
      saveUiPreferences();
    });
    elements.toggleSubBoxFlowText.addEventListener("click", () => {
      if (!state.showSubBoxFlow) {
        return;
      }
      state.showSubBoxFlowText = !state.showSubBoxFlowText;
      refreshCurrentFrameVisuals();
      updateLabels();
      saveUiPreferences();
    });
    elements.toggleStaticLabels.addEventListener("click", () => {
      if (!state.meta?.hasStaticLabels || state.showLossView) {
        return;
      }
      state.showStaticLabels = !state.showStaticLabels;
      applyCurrentPointVisuals();
      renderStatusRows(buildStatusRows());
      updateLabels();
      saveUiPreferences();
    });
    elements.toggleHideLossBelowThreshold.addEventListener("click", () => {
      if (!state.showLossView || !hasCurrentLossData()) {
        return;
      }
      state.hideLossBelowThreshold = !state.hideLossBelowThreshold;
      applyCurrentPointVisuals();
      renderStatusRows(buildStatusRows());
      updateLabels();
      saveUiPreferences();
    });
    elements.toggleHideIgnorePoints.addEventListener("click", () => {
      if (!state.showLossView || !hasCurrentLossData()) {
        return;
      }
      state.hideIgnorePoints = !state.hideIgnorePoints;
      applyCurrentPointVisuals();
      renderStatusRows(buildStatusRows());
      updateLabels();
      saveUiPreferences();
    });
    elements.lossRatioSlider.addEventListener("input", (event) => {
      state.lossRatioPercent = clampLossRatioPercent(Number(event.target.value));
      if (state.currentLossStats) {
        state.lossThreshold = getLossThresholdFromPercentile(state.currentLossStats, state.lossRatioPercent);
      }
      applyCurrentPointVisuals();
      renderStatusRows(buildStatusRows());
      updateLabels();
      saveUiPreferences();
    });
    elements.lossRatioInput.addEventListener("change", (event) => {
      state.lossRatioPercent = clampLossRatioPercent(Number(event.target.value));
      if (state.currentLossStats) {
        state.lossThreshold = getLossThresholdFromPercentile(state.currentLossStats, state.lossRatioPercent);
      }
      applyCurrentPointVisuals();
      renderStatusRows(buildStatusRows());
      updateLabels();
      saveUiPreferences();
      updateLabels();
      saveUiPreferences();
    });
    elements.lossThresholdInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && state.currentLossRange) {
        state.lossThreshold = clampLossThreshold(Number(event.target.value), state.currentLossRange);
        state.lossRatioPercent = getLossPercentileFromThreshold(state.currentLossStats, state.lossThreshold);
        applyCurrentPointVisuals();
        renderStatusRows(buildStatusRows());
        updateLabels();
        saveUiPreferences();
      }
    });
    elements.lossRatioInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        state.lossRatioPercent = clampLossRatioPercent(Number(event.target.value));
        if (state.currentLossStats) {
          state.lossThreshold = getLossThresholdFromPercentile(state.currentLossStats, state.lossRatioPercent);
        }
        applyCurrentPointVisuals();
        renderStatusRows(buildStatusRows());
        updateLabels();
      }
    });
    elements.lossThresholdSlider.addEventListener("input", (event) => {
      if (!state.currentLossRange) {
        return;
      }
      state.lossThreshold = clampLossThreshold(Number(event.target.value), state.currentLossRange);
      state.lossRatioPercent = getLossPercentileFromThreshold(state.currentLossStats, state.lossThreshold);
      applyCurrentPointVisuals();
      renderStatusRows(buildStatusRows());
      updateLabels();
    });
    elements.lossThresholdInput.addEventListener("change", (event) => {
      if (!state.currentLossRange) {
        return;
      }
      state.lossThreshold = clampLossThreshold(Number(event.target.value), state.currentLossRange);
      state.lossRatioPercent = getLossPercentileFromThreshold(state.currentLossStats, state.lossThreshold);
      applyCurrentPointVisuals();
      renderStatusRows(buildStatusRows());
      updateLabels();
    });
    elements.lossThresholdInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && state.currentLossRange) {
        state.lossThreshold = clampLossThreshold(Number(event.target.value), state.currentLossRange);
        state.lossRatioPercent = getLossPercentileFromThreshold(state.currentLossStats, state.lossThreshold);
        applyCurrentPointVisuals();
        renderStatusRows(buildStatusRows());
        updateLabels();
      }
    });
    elements.detLineThicknessInput.addEventListener("input", (event) => {
      setDetLineThickness(Number(event.target.value));
    });
    elements.detScoreSlider.addEventListener("input", (event) => {
      setDetScoreThreshold(Number(event.target.value));
    });
    elements.detScoreInput.addEventListener("change", (event) => {
      setDetScoreThreshold(Number(event.target.value));
    });
    elements.detScoreInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        setDetScoreThreshold(Number(event.target.value));
      }
    });
    elements.bboxPanelClose.addEventListener("click", () => {
      setSelectedDetection(null);
    });
    elements.modeGt.addEventListener("click", () => setViewMode("gt"));
    elements.modePred.addEventListener("click", () => setViewMode("pred"));
    elements.modeLoss.addEventListener("click", () => setViewMode("loss"));
    elements.modeEdit.addEventListener("click", () => setViewMode("edit"));
    elements.clearEditCurrent.addEventListener("click", () => {
      const frameKey = getCurrentFrameKey();
      if (!frameKey) {
        return;
      }
      delete state.editRectangles[frameKey];
      state.editDraftStart = null;
      renderEditPreviewRectangle(null);
      renderEditCurrentRectangle(null);
      appendLog(`Rectangle cleared for ${frameKey}.`);
      renderStatusRows(buildStatusRows());
      updateLabels();
    });
    elements.saveEditRectangles.addEventListener("click", async () => {
      try {
        const response = await fetch(appUrl("api/save_rectangles"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rectangles: state.editRectangles }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Failed to save rectangles");
        }
        appendLog(`Rectangles saved: ${payload.count} frame(s) -> ${payload.path}`);
        renderStatusRows(buildStatusRows());
        updateLabels();
      } catch (error) {
        appendLog(`Rectangle save failed: ${error.message}`, "error");
      }
    });
    elements.openImagePanel.addEventListener("click", () => {
      openImageFloatingPanel();
    });
    elements.addImagePanelWindow.addEventListener("click", () => {
      addImagePanelWindow(elements.imagePanelCameraSelect.value || state.imagePanelDefaultCameraKey);
    });
    elements.imagePanelCameraSelect.addEventListener("change", (event) => {
      state.imagePanelDefaultCameraKey = event.target.value || "";
      syncImagePanelSelectors();
      saveUiPreferences();
    });
    elements.lossHistogramClose.addEventListener("click", () => {
      state.lossHistogramDismissed = true;
      updateLabels();
    });
    elements.logPanelMinimize.addEventListener("click", () => {
      toggleLogPanel();
    });
    elements.scanRoot.addEventListener("click", () => {
      loadSelectedDir();
    });
    elements.loadSelected.addEventListener("click", () => {
      loadSelectedDir();
    });
    elements.frameSlider.addEventListener("input", (event) => {
      showFrame(Number(event.target.value), { force: true });
    });
    elements.jumpFrame.addEventListener("click", () => {
      jumpToFrameInput();
    });
    elements.frameJumpInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        jumpToFrameInput();
      }
    });
    elements.jumpPercent.addEventListener("click", () => {
      jumpToPercentInput();
    });
    elements.framePercentInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        jumpToPercentInput();
      }
    });
    elements.jumpRandom.addEventListener("click", () => {
      jumpToRandomFrame();
    });
    elements.fpsSlider.addEventListener("input", (event) => {
      state.fps = Number(event.target.value);
      updateLabels();
      saveUiPreferences();
    });
    elements.pointSizeSlider.addEventListener("input", (event) => {
      state.pointSize = Number(event.target.value);
      if (state.currentMesh) {
        state.currentMesh.material.size = state.pointSize;
      }
      if (state.currentPrevMesh) {
        state.currentPrevMesh.material.size = Math.max(1, state.pointSize);
      }
      updateLabels();
      saveUiPreferences();
    });

    window.addEventListener("keydown", (event) => {
      const tagName = event.target?.tagName;
      const isTypingTarget = tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT";
      if (!isTypingTarget && event.code === "ArrowRight") {
        showFrame(state.index + 1, { force: true });
      } else if (!isTypingTarget && event.code === "ArrowLeft") {
        showFrame(state.index - 1, { force: true });
      }
    });
    renderer.domElement.addEventListener("pointerdown", (event) => {
      if (focusViewOnPointer(event)) {
        state.pointerDown = null;
        event.preventDefault();
        return;
      }
      state.pointerDown = { x: event.clientX, y: event.clientY };
    });
    renderer.domElement.addEventListener("pointermove", (event) => {
      if (!state.editMode || !state.editDraftStart) {
        return;
      }
      const groundPoint = getGroundPointFromPointerEvent(event);
      if (!groundPoint) {
        return;
      }
      renderEditPreviewRectangle(rectangleFromCorners(state.editDraftStart, groundPoint));
    });
    renderer.domElement.addEventListener("pointerup", (event) => {
      if (!state.pointerDown) {
        return;
      }
      if (event.button !== 0) {
        state.pointerDown = null;
        return;
      }
      const dx = event.clientX - state.pointerDown.x;
      const dy = event.clientY - state.pointerDown.y;
      state.pointerDown = null;
      if (dx * dx + dy * dy > 25) {
        return;
      }
      handleCanvasClick(event);
    });
    elements.rootDirInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        loadSelectedDir();
      }
    });
    elements.rootDirInput.addEventListener("input", (event) => {
      refreshPathEntries(event.target.value, "pkl_file", elements.pklPathEntries);
    });
    elements.subdirSelect.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        loadSelectedDir();
      }
    });
    elements.subdirSelect.addEventListener("input", (event) => {
      refreshPathEntries(event.target.value, "eval_dir", elements.evalDirEntries);
    });
    elements.pklPathEntries.addEventListener("change", (event) => {
      applyPathEntry(elements.rootDirInput, elements.pklPathEntries, event.target.selectedOptions[0], "pkl_file");
    });
    elements.pklPathEntries.addEventListener("dblclick", (event) => {
      applyPathEntry(elements.rootDirInput, elements.pklPathEntries, event.target.selectedOptions[0], "pkl_file");
    });
    elements.evalDirEntries.addEventListener("change", (event) => {
      applyPathEntry(elements.subdirSelect, elements.evalDirEntries, event.target.selectedOptions[0], "eval_dir");
    });
    elements.evalDirEntries.addEventListener("dblclick", (event) => {
      applyPathEntry(elements.subdirSelect, elements.evalDirEntries, event.target.selectedOptions[0], "eval_dir");
    });

    window.addEventListener("error", (event) => {
      appendLog(`Unhandled error: ${event.error?.message || event.message || "Unknown error"}`, "error");
    });
    window.addEventListener("unhandledrejection", (event) => {
      const reason = event.reason;
      appendLog(
        `Unhandled rejection: ${reason?.message || String(reason || "Unknown rejection")}`,
        "error"
      );
    });

    function animate(nowMs) {
      requestAnimationFrame(animate);
      if (state.meta && state.playing && !state.loading) {
        const intervalMs = 1000 / Math.max(state.fps, 1);
        if (nowMs - state.lastTickMs >= intervalMs) {
          state.lastTickMs = nowMs;
          if (state.index >= state.meta.frameCount - 1) {
            state.playing = false;
            updateLabels();
          } else {
            showFrame(state.index + 1);
          }
        }
      } else {
        state.lastTickMs = nowMs;
      }
      controls.update();
      renderer.render(scene, camera);
    }

    init().catch((error) => {
      renderStatusRows([{ key: "Error", value: `Failed to load viewer: ${error.message}` }]);
      console.error(error);
      appendLog(`Viewer init failed: ${error.message}`, "error");
    });
    updateLabels();
    animate(performance.now());
  </script>
</body>
</html>
"""


def make_handler(
    app_state: ViewerState,
    fps: float,
    point_size: float,
    sample_rate: int,
    label_source: str,
    at720: bool,
):
    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(
            self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, obj: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send_bytes(
                json.dumps(obj).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            payload = self.rfile.read(length)
            return json.loads(payload.decode("utf-8"))

        def log_message(self, fmt: str, *args) -> None:
            sys.stdout.write("[http] " + fmt % args + "\n")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/static/three.module.js":
                self._send_bytes(
                    (VENDOR_DIR / "three.module.js").read_bytes(),
                    "application/javascript; charset=utf-8",
                )
                return
            if parsed.path == "/static/OrbitControls.js":
                orbit_controls = (VENDOR_DIR / "OrbitControls.js").read_text(encoding="utf-8")
                orbit_controls = orbit_controls.replace(
                    "from '/static/three.module.js';",
                    "from './three.module.js';",
                )
                self._send_bytes(
                    orbit_controls.encode("utf-8"),
                    "application/javascript; charset=utf-8",
                )
                return
            if parsed.path == "/api/meta":
                store = app_state.get_store()
                meta = store.meta(fps=fps, point_size=point_size)
                self._send_json(meta)
                return
            if parsed.path == "/api/path_suggestions":
                try:
                    query = parse_qs(parsed.query)
                    prefix = query.get("prefix", [""])[0]
                    kind = query.get("kind", ["pkl_file"])[0]
                    suggestions = app_state.suggest_paths(prefix, kind)
                    self._send_json({"suggestions": suggestions})
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path == "/api/path_entries":
                try:
                    query = parse_qs(parsed.query)
                    path_value = query.get("path", [""])[0]
                    kind = query.get("kind", ["pkl_file"])[0]
                    payload = app_state.list_path_entries(path_value, kind)
                    self._send_json(payload)
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path.startswith("/api/frame_images/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    self._send_json({"items": store.load_frame_images_meta(index)})
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path.startswith("/api/image/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    camera_key = parse_qs(parsed.query).get("camera_key", [""])[0]
                    payload = store.load_frame_image(index, camera_key)
                    if payload is None:
                        self._send_json(
                            {"error": "Image not found"},
                            status=HTTPStatus.NOT_FOUND,
                        )
                        return
                    image_bytes, content_type = payload
                    self._send_bytes(image_bytes, content_type)
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path.startswith("/api/frame/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    label_source = parse_qs(parsed.query).get("labelSource", [None])[0]
                    payload = store.load_frame(index, label_source=label_source)
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._send_bytes(payload, "application/octet-stream")
                return
            if parsed.path.startswith("/api/prev_points/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    payload = store.load_previous_points(index)
                    if payload is None:
                        self._send_json({"error": "Previous points not found"}, status=HTTPStatus.NOT_FOUND)
                        return
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._send_bytes(payload, "application/octet-stream")
                return
            if parsed.path.startswith("/api/det/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    label_source = parse_qs(parsed.query).get("labelSource", [None])[0]
                    self._send_json({"detections": store.load_detections(index, label_source=label_source)})
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path.startswith("/api/frame_info/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    label_source = parse_qs(parsed.query).get("labelSource", [None])[0]
                    self._send_json(store.load_frame_log_info(index, label_source=label_source))
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path.startswith("/api/flow/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    label_source = parse_qs(parsed.query).get("labelSource", [None])[0]
                    payload = store.load_flow(index, label_source=label_source)
                    if payload is None:
                        self._send_json({"error": "Flow not found"}, status=HTTPStatus.NOT_FOUND)
                        return
                    self._send_bytes(payload, "application/octet-stream")
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path.startswith("/api/sub_box_flow/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    label_source = parse_qs(parsed.query).get("labelSource", [None])[0]
                    self._send_json({"items": store.load_sub_box_flows(index, label_source=label_source)})
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path.startswith("/api/static/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    label_source = parse_qs(parsed.query).get("labelSource", [None])[0]
                    payload = store.load_static_labels(index, label_source=label_source)
                    if payload is None:
                        self._send_json({"error": "Static labels not found"}, status=HTTPStatus.NOT_FOUND)
                        return
                    self._send_bytes(payload, "application/octet-stream")
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path.startswith("/api/loss/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    payload = store.load_point_loss(index)
                    if payload is None:
                        self._send_json({"error": "Point loss not found"}, status=HTTPStatus.NOT_FOUND)
                        return
                    self._send_bytes(payload, "application/octet-stream")
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                return
            if parsed.path.startswith("/api/ignore/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    payload = store.load_ignore_mask(index)
                    self._send_bytes(payload, "application/octet-stream")
                except Exception as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/open":
                    payload = self._read_json()
                    pkl_file = payload.get("pklFile", "")
                    eval_dir = payload.get("evalDir", "")
                    if not pkl_file:
                        raise ValueError("Missing pklFile")
                    store = app_state.open_store(
                        pkl_file=pkl_file,
                        eval_dir=eval_dir,
                        sample_rate=sample_rate,
                        label_source=label_source,
                        at720=at720,
                    )
                    meta = store.meta(fps=fps, point_size=point_size)
                    self._send_json({"meta": meta})
                    return
                if parsed.path == "/api/save_rectangles":
                    payload = self._read_json()
                    result = save_edit_rectangles(
                        DEFAULT_EDIT_RECTANGLES_PATH,
                        payload.get("rectangles", {}),
                    )
                    self._send_json(result)
                    return
                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json(
                    {"error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )

    return Handler


def main() -> None:
    args = parse_args()
    store = FrameStore(
        pkl_file=args.pkl_file,
        eval_dir=args.eval_dir,
        sample_rate=args.sample_rate,
        label_source=args.label_source,
        at720=args.at720,
    )
    app_state = ViewerState(store)
    handler = make_handler(
        app_state,
        fps=args.fps,
        point_size=args.point_size,
        sample_rate=args.sample_rate,
        label_source=args.label_source,
        at720=args.at720,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"

    print(f"Serving {len(store)} frames from {store.pkl_file}")
    if store.eval_dir is not None:
        print(f"Eval dir: {store.eval_dir}")
    print(f"Open {url}")
    print("Keyboard in browser: Space play/pause, Left/Right step frames")

    if args.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Server stopped")


if __name__ == "__main__":
    main()
