"""Internal ADRN-based frame bundle loader for Xiaomi datasets."""

from __future__ import annotations

import os
import pickle
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from ad_cloud.adrn.data_seeker.frame import (
    _get_local_path,
    frame_adrn_to_cache_path,
    read_frame,
)


RAW_FRAME_CACHE_PATH = "/high_perf_store3/ad-cloud-data-cache/frame/"
PC_RANGE = (-24.0, -80.0, -10.0, 24.0, 200.0, 10.0)
PT_SEG_MAP_DICT = np.array(
    [
        4, 12, 12, 12, 12, 0, 12, 12, 12, 13,
        4, 6, 9, 2, 2, 9, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 4, 2, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 9,
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
}

RENDER_CLASSNAME_TO_COLOR = np.array(
    [
        (255, 255, 255),
        (255, 99, 71),
        (255, 20, 147),
        (0, 191, 255),
        (84, 255, 159),
        (65, 105, 225),
        (47, 79, 79),
        (255, 255, 0),
        (255, 140, 0),
        (255, 0, 255),
        (255, 218, 185),
        (0, 255, 255),
        (188, 143, 143),
        (128, 128, 128),
    ],
    dtype=np.uint8,
)


def simple_read_frame(adrn: str, mount_path: str | None = None):
    if mount_path is not None:
        return read_frame(adrn, mount_path)

    raw_path = _get_local_path(frame_adrn_to_cache_path(adrn), RAW_FRAME_CACHE_PATH)
    if os.path.exists(raw_path):
        return read_frame(adrn, RAW_FRAME_CACHE_PATH)
    return None


def _decode_single_frame(adrn: str) -> np.ndarray | None:
    lidar_bytes = simple_read_frame(adrn)
    if lidar_bytes is None:
        return None
    lidar_bytes = bytes(lidar_bytes)
    return np.frombuffer(lidar_bytes, dtype=np.float32, count=-1).reshape(-1, 6)


def _normalize_name(name: Any) -> str:
    return CATEMAP.get(str(name), str(name))


def _map_raw_gt_labels(raw_labels: np.ndarray) -> np.ndarray:
    raw_labels = np.asarray(raw_labels).reshape(-1).astype(np.int32, copy=False)
    mapped = np.full(raw_labels.shape, 12, dtype=np.int32)
    valid = (raw_labels >= 0) & (raw_labels < len(PT_SEG_MAP_DICT))
    mapped[valid] = PT_SEG_MAP_DICT[raw_labels[valid]]
    return mapped


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


def _ensure_2d_array(values: np.ndarray, min_cols: int) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        return np.empty((0, min_cols), dtype=np.float32)
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    if array.ndim != 2 or array.shape[1] < min_cols:
        raise ValueError(f"Unexpected array shape: {array.shape}")
    return array


def _prediction_path(eval_dir: Path | None, data_info: dict[str, Any]) -> Path | None:
    if eval_dir is None:
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
    return eval_dir / (str(adrn_value).replace(":", "_") + ".pkl")


def _build_gt_detections(data_info: dict[str, Any]) -> list[dict]:
    gt_boxes = data_info.get("gt_boxes")
    gt_names = data_info.get("gt_names")
    if gt_boxes is None or gt_names is None:
        return []
    boxes = _ensure_2d_array(gt_boxes, 7).astype(np.float32)
    detections = []
    for row, raw_name in zip(boxes, list(gt_names)):
        name = _normalize_name(raw_name)
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


def _build_pred_detections(pred_data: dict[str, Any] | None) -> list[dict]:
    if not pred_data:
        return []
    voxel_results = pred_data.get("voxel_results")
    if not isinstance(voxel_results, dict):
        return []
    if "boxes_3d" not in voxel_results or "labels_3d" not in voxel_results:
        return []
    boxes = _ensure_2d_array(voxel_results["boxes_3d"], 7).astype(np.float32)
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


def load_points(lidar_source: Any, at720: bool = False):
    try:
        if isinstance(lidar_source, list):
            if at720:
                total_data = {"points": [], "names": []}
                for source in lidar_source:
                    if "lidar" not in source:
                        continue
                    lidar_points = _decode_single_frame(source)
                    if lidar_points is None:
                        continue
                    total_data["points"].append(lidar_points)
                    total_data["names"].append(source)
                lidar_points = total_data
            else:
                total_data = []
                for source in lidar_source:
                    lidar_points = _decode_single_frame(source)
                    if lidar_points is None:
                        continue
                    if "left" in source or "right" in source:
                        lidar_points = lidar_points.copy()
                        lidar_points[:, 3] = lidar_points[:, 3] / 4096 * 25.5
                    total_data.append(lidar_points)
                if not total_data:
                    return None, None
                lidar_points = np.concatenate(total_data, axis=0)
        elif isinstance(lidar_source, dict):
            total_data = []
            for key, source in lidar_source.items():
                lidar_points = _decode_single_frame(source)
                if lidar_points is None:
                    continue
                if key != "Major":
                    lidar_points = lidar_points.copy()
                    lidar_points[:, 3] = lidar_points[:, 3] / 4096 * 25.5
                total_data.append(lidar_points)
            if not total_data:
                return None, None
            lidar_points = np.concatenate(total_data, axis=0)
        else:
            lidar_points = _decode_single_frame(lidar_source)
            if lidar_points is None:
                return None, None

        if isinstance(lidar_points, dict):
            if all("lidar.mid_center_top_wide" not in name for name in lidar_points["names"]):
                return None, None
            total_data = []
            total_flag = []
            for idx, name in enumerate(lidar_points["names"]):
                if "lidar.mid_center_top_wide" in name:
                    at720_points = lidar_points["points"][idx]
                    mask1 = (
                        (np.abs(at720_points[:, 0]) < 1000)
                        & (np.abs(at720_points[:, 1]) < 1000)
                        & (np.abs(at720_points[:, 2]) < 1000)
                    )
                    lidar_xy_norm = np.linalg.norm(at720_points[:, :2], axis=1)
                    mask2 = lidar_xy_norm >= 4.0
                    final_valid_mask = mask1 & mask2
                    total_data.append(at720_points[final_valid_mask])
                    total_flag.append(final_valid_mask)
                else:
                    other_points = lidar_points["points"][idx]
                    mask = (
                        (np.abs(other_points[:, 0]) < 1000)
                        & (np.abs(other_points[:, 1]) < 1000)
                        & (np.abs(other_points[:, 2]) < 1000)
                    )
                    total_data.append(other_points[mask])
                    total_flag.append(mask)
            if not total_data:
                return None, None
            merged_points = np.concatenate(total_data, axis=0)
            merged_mask = np.concatenate(total_flag, axis=0)
            return merged_points, merged_mask

        mask = (
            (np.abs(lidar_points[:, 0]) < 1000)
            & (np.abs(lidar_points[:, 1]) < 1000)
            & (np.abs(lidar_points[:, 2]) < 1000)
        )
        return lidar_points[mask], mask
    except Exception:
        print(f"Failed to load lidar source: {lidar_source}")
        print(traceback.format_exc())
        return None, None


def load_frame_bundle(
    source_file: str | Path,
    *,
    eval_dir: str | Path | None = None,
    at720: bool = False,
) -> dict[str, Any] | None:
    source_path = Path(source_file).expanduser()
    if source_path.suffix.lower() != ".pkl":
        return None

    try:
        with open(source_path, "rb") as handle:
            data_info = pickle.load(handle)

        lidar_source = data_info["adrns"] if "adrns" in data_info else data_info.get("adrn")
        lidar_data, base_mask = load_points(lidar_source, at720=at720)
        if lidar_data is None or base_mask is None:
            raise ValueError(f"Failed to load lidar points for {source_path.name}")

        eval_dir_path = None if not eval_dir else Path(eval_dir).expanduser()
        pred_path = _prediction_path(eval_dir_path, data_info)
        pred_data = None
        if pred_path is not None and pred_path.exists():
            with open(pred_path, "rb") as handle:
                pred_data = pickle.load(handle)

        range_mask = _point_range_mask(lidar_data)
        visible_points = lidar_data[range_mask]
        point_selector = None
        if pred_data and pred_data.get("points_voxel_sample_mask") is not None:
            point_selector = np.asarray(pred_data["points_voxel_sample_mask"]).astype(bool, copy=False)
            if len(point_selector) != len(visible_points):
                raise ValueError(
                    f"points_voxel_sample_mask length mismatch for {source_path.name}: "
                    f"{len(point_selector)} vs {len(visible_points)}"
                )
            visible_points = visible_points[point_selector]

        gt_seg_raw = data_info.get("gt_seg")
        raw_gt_visible = None
        if gt_seg_raw is None:
            gt_labels = np.zeros((len(visible_points),), dtype=np.int32)
        else:
            gt_valid = np.asarray(gt_seg_raw)[np.asarray(base_mask).astype(bool, copy=False)]
            raw_gt_visible = gt_valid[range_mask]
            if point_selector is not None:
                raw_gt_visible = raw_gt_visible[point_selector]
            gt_labels = _map_raw_gt_labels(raw_gt_visible)

        pred_labels = None
        flow_values = None
        static_labels = None
        if pred_data and pred_data.get("pts_results") is not None:
            pred_count = pred_data.get("counts")
            pred_labels = np.asarray(pred_data["pts_results"]).reshape(-1).astype(np.int32, copy=False)
            if pred_count is not None:
                pred_labels = pred_labels[:pred_count]
            if len(pred_labels) != len(visible_points):
                raise ValueError(
                    f"pts_results length mismatch for {source_path.name}: {len(pred_labels)} vs {len(visible_points)}"
                )

            flow_values = pred_data.get("flow_pred")
            if flow_values is not None:
                flow_values = np.asarray(flow_values)
                if pred_count is not None:
                    flow_values = flow_values[:pred_count]

            static_labels = pred_data.get("point_static")
            if static_labels is not None:
                static_labels = np.asarray(static_labels).reshape(-1)
                if pred_count is not None:
                    static_labels = static_labels[:pred_count]

            ignore_mask = gt_labels != 13
            visible_points = visible_points[ignore_mask]
            gt_labels = gt_labels[ignore_mask]
            pred_labels = pred_labels[ignore_mask]
            if flow_values is not None:
                flow_values = flow_values[ignore_mask]
            if static_labels is not None:
                static_labels = static_labels[ignore_mask]
        elif data_info.get("flow_gt") is not None:
            flow_values = np.asarray(data_info["flow_gt"])[range_mask]
            if point_selector is not None:
                flow_values = flow_values[point_selector]

        gt_detections = _build_gt_detections(data_info)
        pred_detections = _build_pred_detections(pred_data)
        return {
            "name": source_path.name,
            "points": np.ascontiguousarray(visible_points[:, :3], dtype=np.float32),
            "gt_labels": np.ascontiguousarray(gt_labels.astype(np.int32, copy=False)),
            "pred_labels": None
            if pred_labels is None
            else np.ascontiguousarray(pred_labels.astype(np.int32, copy=False)),
            "flow": None
            if flow_values is None
            else np.ascontiguousarray(flow_values[:, :3], dtype=np.float32),
            "static_labels": None
            if static_labels is None
            else np.ascontiguousarray(np.clip(static_labels.astype(np.int32), 0, 1).astype(np.uint8)),
            "gt_detections": gt_detections,
            "pred_detections": pred_detections,
            "log_info": {
                "entries": [
                    {"key": "mode_default", "value": "gt"},
                    {"key": "frame_file_name", "value": source_path.name},
                    {"key": "frame_file_path", "value": str(source_path)},
                    {"key": "ann_file", "value": data_info.get("anno_file") or data_info.get("clip_name")},
                    {"key": "adrn", "value": data_info.get("adrn")},
                    {"key": "eval_file", "value": None if pred_path is None else str(pred_path)},
                    {"key": "visible_points", "value": len(visible_points)},
                    {
                        "key": "unique_gt_raw",
                        "value": [] if raw_gt_visible is None else np.unique(raw_gt_visible.astype(np.int32)).tolist(),
                    },
                    {"key": "unique_gt_mapped", "value": np.unique(gt_labels.astype(np.int32)).tolist()},
                    {"key": "gt_det_count", "value": len(gt_detections)},
                    {"key": "eval_det_count", "value": len(pred_detections)},
                    {"key": "has_flow", "value": flow_values is not None},
                    {"key": "has_static_labels", "value": static_labels is not None},
                ]
            },
        }
    except Exception:
        print(f"Failed to load frame bundle from {source_path}")
        print(traceback.format_exc())
        return None
