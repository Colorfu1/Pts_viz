#!/usr/bin/env python3
"""
Browser point cloud viewer for PKL-indexed frame datasets.

The top-level PKL is treated as a lightweight frame index. Actual per-frame
pickles, lidar points, labels, detections, flow, and static labels are loaded
only when the browser requests a specific frame.
"""

from __future__ import annotations

import argparse
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

SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = SCRIPT_DIR / "vendor" / "browser_viewer"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "viewer.yaml"


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
}

BBOX_CLASS_NAMES = ["car", "truck", "bus", "pedestrian", "cyclist", "barrier"]
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


def load_points_for_frame(lidar_adrn: Any, at720: bool) -> tuple[np.ndarray | None, np.ndarray | None]:
    from .point_loader import load_points

    return load_points(lidar_adrn, at720=at720)


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
        self.has_flow_modality, self.has_static_modality = self._detect_optional_modalities()

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
        sample_rate: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if sample_rate <= 1:
            return positions, gt_labels, pred_labels, flow, static_labels
        positions = positions[::sample_rate]
        gt_labels = gt_labels[::sample_rate]
        if pred_labels is not None:
            pred_labels = pred_labels[::sample_rate]
        if flow is not None:
            flow = flow[::sample_rate]
        if static_labels is not None:
            static_labels = static_labels[::sample_rate]
        return positions, gt_labels, pred_labels, flow, static_labels

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

    def _build_gt_detections(self, data_info: dict[str, Any]) -> list[dict]:
        gt_boxes = data_info.get("gt_boxes")
        gt_names = data_info.get("gt_names")
        if gt_boxes is None or gt_names is None:
            return []
        boxes = self._ensure_2d_array(gt_boxes, "gt_boxes", 7)
        names = list(gt_names)
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
                }
            )
        return detections

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
        gt_det_count: int,
        pred_det_count: int,
        flow_values: np.ndarray | None,
        static_labels: np.ndarray | None,
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
            {"key": "visible_points", "value": visible_point_count},
            {"key": "unique_gt_raw", "value": raw_unique},
            {"key": "unique_gt_mapped", "value": mapped_unique},
            {"key": "gt_det_count", "value": gt_det_count},
            {"key": "eval_det_count", "value": pred_det_count},
            {"key": "has_flow", "value": flow_values is not None},
            {"key": "has_static_labels", "value": static_labels is not None},
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

    def _detect_optional_modalities(self, max_checks: int = 16) -> tuple[bool, bool]:
        has_flow = False
        has_static = False
        if self.eval_dir is None:
            return has_flow, has_static
        for frame_path in self.frame_paths[:max_checks]:
            try:
                with open(frame_path, "rb") as handle:
                    data_info = pickle.load(handle)
                pred_path = self._prediction_path(data_info)
                if pred_path is None or not pred_path.exists():
                    continue
                with open(pred_path, "rb") as handle:
                    pred_data = pickle.load(handle)
                has_flow = has_flow or ("flow_pred" in pred_data)
                has_static = has_static or ("point_static" in pred_data)
                if has_flow and has_static:
                    break
            except Exception:
                continue
        return has_flow, has_static

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

    @lru_cache(maxsize=2)
    def _load_bundle(self, index: int) -> dict[str, Any]:
        frame_info_path = self._frame_info_path(index)
        with open(frame_info_path, "rb") as handle:
            data_info = pickle.load(handle)

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
        range_mask = self._point_range_mask(selected_points)
        visible_points = selected_points[range_mask]

        gt_seg_raw = data_info.get("gt_seg")
        raw_gt_visible = None
        if gt_seg_raw is None:
            gt_labels = np.zeros((len(visible_points),), dtype=np.int32)
        else:
            raw_gt_visible = np.asarray(gt_seg_raw)[point_selector][range_mask]
            gt_labels = self._map_raw_gt_labels(raw_gt_visible)

        pred_count = None
        pred_labels = None
        flow_values = None
        static_labels = None
        if pred_data and pred_data.get("pts_results") is not None:
            pred_count = pred_data.get("counts")
            pred_labels = self._align_pred_array(
                self._normalize_label(pred_data["pts_results"]),
                pred_count,
                len(selected_points),
                range_mask,
                "pts_results",
            )
            flow_values = self._align_pred_array(
                pred_data.get("flow_pred"),
                pred_count,
                len(selected_points),
                range_mask,
                "flow_pred",
            )
            static_labels = self._align_pred_array(
                pred_data.get("point_static"),
                pred_count,
                len(selected_points),
                range_mask,
                "point_static",
            )
            ignore_mask = gt_labels != 13
            visible_points = visible_points[ignore_mask]
            gt_labels = gt_labels[ignore_mask]
            pred_labels = pred_labels[ignore_mask]
            if flow_values is not None:
                flow_values = flow_values[ignore_mask]
            if static_labels is not None:
                static_labels = static_labels[ignore_mask]
        elif data_info.get("flow_gt") is not None:
            flow_values = np.asarray(data_info["flow_gt"])[point_selector][range_mask]

        positions = np.ascontiguousarray(visible_points[:, :3], dtype=np.float32)
        gt_labels = np.ascontiguousarray(gt_labels.astype(np.int32, copy=False))
        pred_labels = (
            None
            if pred_labels is None
            else np.ascontiguousarray(pred_labels.astype(np.int32, copy=False))
        )
        flow_values = None if flow_values is None else np.ascontiguousarray(flow_values[:, :3], dtype=np.float32)
        static_labels = (
            None
            if static_labels is None
            else np.ascontiguousarray(np.clip(static_labels.astype(np.int32), 0, 1).astype(np.uint8))
        )
        positions, gt_labels, pred_labels, flow_values, static_labels = self._apply_sample_rate(
            positions,
            gt_labels,
            pred_labels,
            flow_values,
            static_labels,
            self.sample_rate,
        )
        gt_detections = self._build_gt_detections(data_info)
        pred_detections = self._build_pred_detections(pred_data)
        return {
            "name": frame_info_path.name,
            "positions": positions,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
            "gt_detections": gt_detections,
            "pred_detections": pred_detections,
            "flow": flow_values,
            "static_labels": static_labels,
            "log_info": self._build_frame_log_info(
                frame_info_path=frame_info_path,
                data_info=data_info,
                pred_path=pred_path,
                raw_gt_visible=raw_gt_visible,
                mapped_gt_visible=gt_labels,
                visible_point_count=len(positions),
                gt_det_count=len(gt_detections),
                pred_det_count=len(pred_detections),
                flow_values=flow_values,
                static_labels=static_labels,
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

    @lru_cache(maxsize=16)
    def load_detections(self, index: int, label_source: str | None = None) -> list[dict]:
        bundle = self._load_bundle(index)
        label_source = self._normalize_label_source(label_source, self.label_source)
        if label_source == "pred" and bundle["pred_detections"]:
            return bundle["pred_detections"]
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
    def load_flow(self, index: int) -> bytes | None:
        flow = self._load_bundle(index)["flow"]
        if flow is None:
            return None
        return self._pack_vectors(b"FLW0", flow[:, :3])

    @lru_cache(maxsize=8)
    def load_static_labels(self, index: int) -> bytes | None:
        static_labels = self._load_bundle(index)["static_labels"]
        if static_labels is None:
            return None
        return self._pack_labels(b"STA0", static_labels)

    def meta(self, fps: float, point_size: float) -> dict:
        return {
            "frameCount": len(self.frame_paths),
            "fps": fps,
            "pointSize": point_size,
            "sampleRate": self.sample_rate,
            "labelSource": self.label_source,
            "at720": self.at720,
            "hasEvalResults": self.eval_dir is not None,
            "hasFlow": self.has_flow_modality,
            "hasStaticLabels": self.has_static_modality,
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
    #switchHint {
      position: absolute;
      right: 18px;
      top: 122px;
      max-width: min(34vw, 360px);
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(2, 8, 23, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.2);
      backdrop-filter: blur(8px);
      font-size: 12px;
      line-height: 1.45;
      display: none;
      align-items: flex-start;
      gap: 10px;
      box-sizing: border-box;
    }
    #switchHint.open {
      display: flex;
    }
    #switchHintText {
      color: var(--text);
      flex: 1 1 auto;
    }
    #switchHintClose {
      border: 0;
      background: rgba(148, 163, 184, 0.14);
      color: var(--text);
      border-radius: 8px;
      width: 24px;
      height: 24px;
      padding: 0;
      font-size: 14px;
      line-height: 1;
      flex: 0 0 auto;
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

      <div class="block">
        <div class="sectionTitle">Data Source</div>
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

      <div class="block">
        <div class="sectionTitle">Playback</div>
        <div class="row">
          <button id="playPause">Pause</button>
          <button class="secondary" id="prevFrame">Prev</button>
          <button class="secondary" id="nextFrame">Next</button>
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

      <div class="block">
        <div class="sectionTitle">OD Boxes</div>
        <div class="row">
          <button class="secondary" id="toggleDetections">OD Boxes: On</button>
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

      <div class="block">
        <div class="sectionTitle">Motion / Static</div>
        <div class="row">
          <button class="secondary" id="toggleFlow">Flow: Off</button>
        </div>
        <div class="row">
          <div style="flex: 1 1 100%;">
            <label for="flowThresholdSlider">Flow Threshold</label>
            <input id="flowThresholdSlider" type="range" min="0" max="5" value="2" step="0.1" />
          </div>
          <div class="value" id="flowThresholdValue">2.0</div>
        </div>
        <div class="row">
          <div style="flex: 1 1 100%;">
            <label for="flowThresholdInput">Flow Threshold Input</label>
            <input id="flowThresholdInput" type="text" value="2.0" />
          </div>
        </div>
        <div class="row">
          <button class="secondary" id="toggleStaticLabels">Static Labels: Off</button>
        </div>
      </div>

      <div class="block">
        <div class="sectionTitle">Status</div>
        <div id="status">Loading metadata...</div>
      </div>

      <div class="block">
        <div class="sectionTitle">Legend</div>
        <div id="legend"></div>
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
      <div id="switchHint">
        <div id="switchHintText"></div>
        <button id="switchHintClose" type="button">×</button>
      </div>
      <div id="bboxPanel">
        <div id="bboxPanelHeader">
          <div id="bboxPanelTitle">Selected OD Box</div>
          <button id="bboxPanelClose" type="button">×</button>
        </div>
        <div id="bboxPanelBody"></div>
      </div>
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
      showFlow: false,
      showStaticLabels: false,
      detLineThickness: 0.12,
      detScoreThreshold: 0.0,
      flowThreshold: 2.0,
      currentMesh: null,
      currentBoxes: null,
      currentSelectionBeam: null,
      currentFlow: null,
      currentFrameData: null,
      currentName: "-",
      currentPoints: 0,
      currentBoxCount: 0,
      labelSource: "gt",
      logMinimized: false,
      hintDismissed: false,
      selectedDetection: null,
      pointerDown: null,
      lastTickMs: performance.now(),
      loading: null,
      subdirs: [],
    };

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
      resetView: document.getElementById("resetView"),
      toggleAutoFit: document.getElementById("toggleAutoFit"),
      toggleDetections: document.getElementById("toggleDetections"),
      toggleFlow: document.getElementById("toggleFlow"),
      flowThresholdSlider: document.getElementById("flowThresholdSlider"),
      flowThresholdValue: document.getElementById("flowThresholdValue"),
      flowThresholdInput: document.getElementById("flowThresholdInput"),
      toggleStaticLabels: document.getElementById("toggleStaticLabels"),
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
      switchHint: document.getElementById("switchHint"),
      switchHintText: document.getElementById("switchHintText"),
      switchHintClose: document.getElementById("switchHintClose"),
      bboxPanel: document.getElementById("bboxPanel"),
      bboxPanelBody: document.getElementById("bboxPanelBody"),
      bboxPanelClose: document.getElementById("bboxPanelClose"),
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
    controls.target.set(30, 0, 0);
    controls.addEventListener("start", () => {
      if (state.autoFit) {
        state.autoFit = false;
        updateLabels();
      }
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

    function updateLabels() {
      const total = state.meta ? state.meta.frameCount : 0;
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
      const flowThresholdText = state.flowThreshold.toFixed(1);
      elements.flowThresholdSlider.value = flowThresholdText;
      elements.flowThresholdValue.textContent = flowThresholdText;
      elements.flowThresholdInput.value = flowThresholdText;
      elements.overlayName.textContent = `Frame: ${state.currentName}`;
      elements.overlayPoints.textContent = `Points: ${state.currentPoints.toLocaleString()}`;
      elements.overlayBboxes.textContent = `OD Boxes: ${state.currentBoxCount.toLocaleString()}`;
      elements.overlayFps.textContent = `Playback: ${state.playing ? `${state.fps} FPS` : "Paused"} | Mode: ${state.labelSource.toUpperCase()}`;
      elements.playPause.textContent = state.playing ? "Pause" : "Play";
      elements.toggleAutoFit.textContent = `Auto Fit: ${state.autoFit ? "On" : "Off"}`;
      elements.toggleDetections.textContent = `OD Boxes: ${state.showDetections ? "On" : "Off"}`;
      elements.toggleFlow.textContent = `Flow: ${state.showFlow ? "On" : "Off"}`;
      elements.toggleStaticLabels.textContent = state.meta?.hasStaticLabels
        ? `Static Labels: ${state.showStaticLabels ? "On" : "Off"}`
        : "Static Labels: N/A";
      elements.toggleStaticLabels.disabled = !state.meta?.hasStaticLabels;
      updateSwitchHint();
    }

    function updateSwitchHint() {
      if (!state.meta || state.hintDismissed) {
        elements.switchHint.classList.remove("open");
        return;
      }
      if (state.meta.hasEvalResults) {
        elements.switchHintText.textContent = "Press T to switch between GT and PRED.";
      } else {
        elements.switchHintText.textContent = "GT only now. Add an eval dir to enable T switching.";
      }
      elements.switchHint.classList.add("open");
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
          appUrl(`api/path_entries?path=${encodeURIComponent(value || "")}&kind=${encodeURIComponent(kind)}`)
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

    function disposeCurrentFlow() {
      if (!state.currentFlow) {
        return;
      }
      state.currentFlow.geometry.dispose();
      state.currentFlow.material.dispose();
      scene.remove(state.currentFlow);
      state.currentFlow = null;
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

    function buildPointColorBuffer(frame, staticLabels) {
      if (!state.showStaticLabels || !staticLabels) {
        return frame.colors;
      }
      const colors = new Uint8Array(frame.pointCount * 3);
      const palette = [
        [255, 255, 255],
        [255, 32, 16],
      ];
      for (let i = 0; i < frame.pointCount; i += 1) {
        const label = Math.min(1, staticLabels.labels[i]);
        const src = i * 3;
        colors[src] = palette[label][0];
        colors[src + 1] = palette[label][1];
        colors[src + 2] = palette[label][2];
      }
      return colors;
    }

    function createFlowOverlay(frame, flowData) {
      disposeCurrentFlow();
      if (!state.showFlow || !flowData) {
        return;
      }

      const linePositions = [];
      const flowThreshold = state.flowThreshold;
      const maxLines = 18000;
      let kept = 0;
      let stride = 1;
      const total = flowData.count;
      if (total > maxLines) {
        stride = Math.ceil(total / maxLines);
      }

      for (let i = 0; i < total; i += stride) {
        const ps = i * 3;
        const vx = flowData.values[ps];
        const vy = flowData.values[ps + 1];
        const vz = flowData.values[ps + 2];
        const flowNorm = Math.hypot(vx, vy);
        if (flowNorm < flowThreshold) {
          continue;
        }
        const x0 = frame.positions[ps];
        const y0 = frame.positions[ps + 1];
        const z0 = frame.positions[ps + 2];
        const x1 = x0 + vx;
        const y1 = y0 + vy;
        const z1 = z0 + vz;
        linePositions.push(x0, z0, -y0, x1, z1, -y1);
        kept += 1;
      }

      if (!linePositions.length) {
        return;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
      const material = new THREE.LineBasicMaterial({
        color: 0xff8c3a,
        transparent: true,
        opacity: 0.8,
        depthWrite: false,
      });
      state.currentFlow = new THREE.LineSegments(geometry, material);
      scene.add(state.currentFlow);
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

    async function fetchFlow(index) {
      if (flowCache.has(index)) {
        return flowCache.get(index);
      }
      const response = await fetch(appUrl(`api/flow/${index}`));
      if (response.status === 404) {
        return null;
      }
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || `Flow request failed: ${response.status}`);
      }
      const parsed = parseVectorBuffer(await response.arrayBuffer(), "FLW0");
      flowCache.set(index, parsed);
      if (flowCache.size > 6) {
        const firstKey = flowCache.keys().next().value;
        flowCache.delete(firstKey);
      }
      return parsed;
    }

    async function fetchStaticLabels(index) {
      if (staticCache.has(index)) {
        return staticCache.get(index);
      }
      const response = await fetch(appUrl(`api/static/${index}`));
      if (response.status === 404) {
        return null;
      }
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || `Static request failed: ${response.status}`);
      }
      const parsed = parseStaticBuffer(await response.arrayBuffer());
      staticCache.set(index, parsed);
      if (staticCache.size > 6) {
        const firstKey = staticCache.keys().next().value;
        staticCache.delete(firstKey);
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
        disposeCurrentMesh();
        disposeCurrentBoxes();
        setSelectedDetection(null);
        state.meta = payload.meta;
        state.index = 0;
        state.playing = false;
        state.labelSource = state.meta.labelSource || "gt";
        state.showStaticLabels = Boolean(state.meta.hasStaticLabels);
        state.currentName = "-";
        state.currentPoints = 0;
        state.currentBoxCount = 0;
        elements.rootDirInput.value = payload.meta.pklFile || "";
        elements.subdirSelect.value = payload.meta.evalDir || "";
        syncPathBrowsers();
        elements.frameSlider.max = String(Math.max(state.meta.frameCount - 1, 0));
        elements.fpsSlider.value = String(Math.round(state.meta.fps));
        elements.pointSizeSlider.value = String(state.meta.pointSize);
        state.fps = Math.round(state.meta.fps);
        state.pointSize = Number(state.meta.pointSize);
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

    function getFilteredDetections(detections) {
      return detections.filter((det) => Number(det.score) >= state.detScoreThreshold);
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
    }

    function createEdgeCylinder(start, end, radius, color) {
      const startVec = new THREE.Vector3(...start);
      const endVec = new THREE.Vector3(...end);
      const delta = new THREE.Vector3().subVectors(endVec, startVec);
      const length = delta.length();
      const geometry = new THREE.CylinderGeometry(radius, radius, Math.max(length, 1e-4), 10);
      const material = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.95,
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

        const edgeColor = new THREE.Color(...det.color.map((value) => value / 255));
        for (let edgeIdx = 0; edgeIdx < linePositions.length; edgeIdx += 6) {
          const edge = createEdgeCylinder(
            linePositions.slice(edgeIdx, edgeIdx + 3),
            linePositions.slice(edgeIdx + 3, edgeIdx + 6),
            state.detLineThickness / 2,
            edgeColor
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
      camera.near = Math.max(0.1, distance / 200);
      camera.far = distance + radius * 6;
      camera.lookAt(center);
      camera.updateProjectionMatrix();
      controls.update();
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

      const promises = [
        fetchFrame(normalized, state.labelSource),
        fetchDetections(normalized, state.labelSource),
        fetchFrameInfo(normalized, state.labelSource),
      ];
      if (state.meta?.hasStaticLabels) {
        promises.push(fetchStaticLabels(normalized));
      } else {
        promises.push(Promise.resolve(null));
      }
      if (state.meta?.hasFlow) {
        promises.push(fetchFlow(normalized));
      } else {
        promises.push(Promise.resolve(null));
      }
      state.loading = Promise.all(promises);
      try {
        const [frame, detections, frameInfo, staticLabels, flowData] = await state.loading;
        state.index = normalized;
        state.currentName = frame.name;
        state.currentPoints = frame.pointCount;
        state.currentFrameData = frame;
        renderFrameLog(frameInfo);

        const geometry = new THREE.BufferGeometry();
        const transformed = new Float32Array(frame.positions.length);
        for (let i = 0; i < frame.pointCount; i += 1) {
          const src = i * 3;
          transformed[src] = frame.positions[src];
          transformed[src + 1] = frame.positions[src + 2];
          transformed[src + 2] = -frame.positions[src + 1];
        }
        geometry.setAttribute("position", new THREE.BufferAttribute(transformed, 3));
        geometry.setAttribute(
          "color",
          new THREE.Uint8BufferAttribute(buildPointColorBuffer(frame, staticLabels), 3, true)
        );

        const material = new THREE.PointsMaterial({
          size: state.pointSize,
          vertexColors: true,
          sizeAttenuation: false,
          transparent: false,
          opacity: 1.0,
          toneMapped: false,
        });

        const points = new THREE.Points(geometry, material);
        const shouldFit = !state.currentMesh || state.autoFit;
        disposeCurrentMesh();
        state.currentMesh = points;
        scene.add(points);
        createBoundingBoxLines(detections);
        createFlowOverlay(frame, flowData);

        if (shouldFit) {
          fitCameraFromGeometry(geometry);
        }

        elements.frameSlider.value = String(state.index);
        renderStatusRows([
          { key: "PKL File", value: state.meta.pklFile },
          { key: "Eval Dir", value: state.meta.evalDir || "-" },
          { key: "Sample Rate", value: `every ${state.meta.sampleRate} point(s)` },
          { key: "Label Source", value: state.labelSource },
        ]);
        updateLabels();
        if (force || !state.playing) {
          appendLog(`Frame ${state.index + 1}/${state.meta.frameCount} loaded in ${state.labelSource.toUpperCase()} mode.`);
        }

        const nextIndex = (state.index + 1) % state.meta.frameCount;
        fetchFrame(nextIndex, state.labelSource).catch(() => {});
        fetchDetections(nextIndex, state.labelSource).catch(() => {});
        fetchFrameInfo(nextIndex, state.labelSource).catch(() => {});
        if (state.meta?.hasStaticLabels) {
          fetchStaticLabels(nextIndex).catch(() => {});
        }
        if (state.meta?.hasFlow) {
          fetchFlow(nextIndex).catch(() => {});
        }
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
      await showFrame(state.index, { force: true });
    }

    async function init() {
      const response = await fetch(appUrl("api/meta"));
      state.meta = await response.json();
      state.labelSource = state.meta.labelSource || "gt";
      state.fps = Math.round(state.meta.fps);
      state.pointSize = Number(state.meta.pointSize);
      elements.rootDirInput.value = state.meta.pklFile || "";
      elements.subdirSelect.value = state.meta.evalDir || "";
      syncPathBrowsers();

      elements.frameSlider.max = String(Math.max(state.meta.frameCount - 1, 0));
      elements.fpsSlider.value = String(state.fps);
      elements.pointSizeSlider.value = String(state.pointSize);
      elements.detScoreSlider.value = String(state.detScoreThreshold);
      state.showStaticLabels = Boolean(state.meta.hasStaticLabels);
      renderLegend(state.meta.classes);
      initResizableLogPanel();
      initDraggableLogPanel();
      initSidebarResizer();
      trapLogPanelEvents();
      syncPathBrowsers();
      renderStatusRows([
        { key: "PKL File", value: state.meta.pklFile || "-" },
        { key: "Eval Dir", value: state.meta.evalDir || "-" },
        { key: "Sample Rate", value: `every ${state.meta.sampleRate} point(s)` },
        { key: "Label Source", value: state.labelSource },
      ]);
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
    elements.resetView.addEventListener("click", () => {
      if (state.currentMesh) {
        fitCameraFromGeometry(state.currentMesh.geometry);
      }
    });
    elements.toggleAutoFit.addEventListener("click", () => {
      state.autoFit = !state.autoFit;
      updateLabels();
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
    });
    elements.toggleFlow.addEventListener("click", () => {
      state.showFlow = !state.showFlow;
      refreshCurrentFrameVisuals();
      updateLabels();
    });
    elements.flowThresholdSlider.addEventListener("input", (event) => {
      state.flowThreshold = Math.min(5.0, Math.max(0.0, Number(event.target.value)));
      refreshCurrentFrameVisuals();
      updateLabels();
    });
    elements.flowThresholdInput.addEventListener("change", (event) => {
      state.flowThreshold = Math.min(5.0, Math.max(0.0, Number(event.target.value)));
      refreshCurrentFrameVisuals();
      updateLabels();
    });
    elements.flowThresholdInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        state.flowThreshold = Math.min(5.0, Math.max(0.0, Number(event.target.value)));
        refreshCurrentFrameVisuals();
        updateLabels();
      }
    });
    elements.toggleStaticLabels.addEventListener("click", () => {
      if (!state.meta?.hasStaticLabels) {
        return;
      }
      state.showStaticLabels = !state.showStaticLabels;
      refreshCurrentFrameVisuals();
      updateLabels();
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
    elements.switchHintClose.addEventListener("click", () => {
      state.hintDismissed = true;
      updateSwitchHint();
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
    });
    elements.pointSizeSlider.addEventListener("input", (event) => {
      state.pointSize = Number(event.target.value);
      if (state.currentMesh) {
        state.currentMesh.material.size = state.pointSize;
      }
      updateLabels();
    });

    window.addEventListener("keydown", (event) => {
      const tagName = event.target?.tagName;
      const isTypingTarget = tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT";
      if (event.code === "Space") {
        event.preventDefault();
        if (!state.playing && state.meta && state.index >= state.meta.frameCount - 1) {
          showFrame(0, { force: true });
        }
        state.playing = !state.playing;
        state.lastTickMs = performance.now();
        updateLabels();
      } else if (!isTypingTarget && event.key.toLowerCase() === "t") {
        if (!state.meta?.hasEvalResults) {
          appendLog("Toggle ignored: no eval results configured.", "warn");
          return;
        }
        state.labelSource = state.labelSource === "gt" ? "pred" : "gt";
        appendLog(`Switched to ${state.labelSource.toUpperCase()} mode.`);
        refreshCurrentFrameVisuals();
        updateLabels();
      } else if (event.code === "ArrowRight") {
        showFrame(state.index + 1, { force: true });
      } else if (event.code === "ArrowLeft") {
        showFrame(state.index - 1, { force: true });
      }
    });
    renderer.domElement.addEventListener("pointerdown", (event) => {
      state.pointerDown = { x: event.clientX, y: event.clientY };
    });
    renderer.domElement.addEventListener("pointerup", (event) => {
      if (!state.pointerDown) {
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
                    payload = store.load_flow(index)
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
            if parsed.path.startswith("/api/static/"):
                try:
                    store = app_state.get_store()
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if index < 0 or index >= len(store):
                        raise IndexError(index)
                    payload = store.load_static_labels(index)
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
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/open":
                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return

            try:
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
