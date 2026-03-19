"""Demo frame bundle loader for repository sample data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEMO_ROOT = Path(__file__).resolve().parent.parent / "examples" / "demo_dataset"
DEMO_BBOX_CLASS_TO_SEG_CLASS = {
    0: 4,
    1: 6,
    2: 5,
    3: 7,
    4: 8,
    5: 12,
    6: 12,
}
DEMO_SEG_COLORS = np.array(
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


def _make_detection(
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
    score: float,
    bbox_class: int,
    seg_class: int,
    name: str,
) -> dict[str, Any]:
    return {
        "center": [float(x) for x in center],
        "size": [float(x) for x in size],
        "yaw": float(yaw),
        "score": float(score),
        "bboxClass": int(bbox_class),
        "segClass": int(seg_class),
        "name": name,
        "color": DEMO_SEG_COLORS[int(seg_class)].tolist(),
    }


def _decode_boxes(
    boxes: np.ndarray,
    bbox_classes: np.ndarray,
    seg_classes: np.ndarray | None,
    scores: np.ndarray | None,
    names: list[str],
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for idx, row in enumerate(np.asarray(boxes)):
        bbox_class = int(np.asarray(bbox_classes)[idx])
        score = 1.0 if scores is None else float(np.asarray(scores)[idx])
        name = names[idx] if idx < len(names) else f"class_{bbox_class}"
        if seg_classes is not None:
            seg_class = int(np.asarray(seg_classes)[idx])
        else:
            seg_class = DEMO_BBOX_CLASS_TO_SEG_CLASS.get(bbox_class, 12)
        detections.append(
            _make_detection(
                center=np.asarray(row[:3], dtype=np.float32),
                size=np.asarray(row[3:6], dtype=np.float32),
                yaw=float(row[6]),
                score=score,
                bbox_class=bbox_class,
                seg_class=seg_class,
                name=name,
            )
        )
    return detections


def load_frame_bundle(
    source_file: str | Path,
    *,
    eval_dir: str | Path | None = None,
    at720: bool = False,
) -> dict[str, Any] | None:
    del eval_dir, at720
    source_path = Path(source_file).expanduser()
    if source_path.suffix.lower() != ".npz":
        return None
    if DEMO_ROOT not in source_path.parents and source_path.parent != DEMO_ROOT:
        return None

    with np.load(source_path, allow_pickle=False) as payload:
        points = np.asarray(payload["points"], dtype=np.float32)
        gt_labels = np.asarray(payload["gt_labels"], dtype=np.int32)
        pred_labels = (
            None
            if "pred_labels" not in payload
            else np.asarray(payload["pred_labels"], dtype=np.int32)
        )
        flow = (
            None
            if "flow" not in payload
            else np.asarray(payload["flow"], dtype=np.float32)
        )
        static_labels = (
            None
            if "static_labels" not in payload
            else np.asarray(payload["static_labels"], dtype=np.uint8)
        )
        gt_boxes = np.asarray(payload["gt_boxes"], dtype=np.float32)
        gt_bbox_classes = np.asarray(payload["gt_bbox_classes"], dtype=np.int32)
        pred_boxes = np.asarray(payload["pred_boxes"], dtype=np.float32)
        pred_bbox_classes = np.asarray(payload["pred_bbox_classes"], dtype=np.int32)
        pred_scores = np.asarray(payload["pred_scores"], dtype=np.float32)
        gt_seg_classes = (
            None
            if "gt_seg_classes" not in payload
            else np.asarray(payload["gt_seg_classes"], dtype=np.int32)
        )
        pred_seg_classes = (
            None
            if "pred_seg_classes" not in payload
            else np.asarray(payload["pred_seg_classes"], dtype=np.int32)
        )
        gt_names = (
            [str(x) for x in payload["gt_names"].tolist()]
            if "gt_names" in payload
            else [f"gt_{idx}" for idx in gt_bbox_classes.tolist()]
        )
        pred_names = (
            [str(x) for x in payload["pred_names"].tolist()]
            if "pred_names" in payload
            else [f"pred_{idx}" for idx in pred_bbox_classes.tolist()]
        )

    gt_detections = _decode_boxes(gt_boxes, gt_bbox_classes, gt_seg_classes, None, gt_names)
    pred_detections = _decode_boxes(pred_boxes, pred_bbox_classes, pred_seg_classes, pred_scores, pred_names)

    return {
        "name": source_path.name,
        "points": points,
        "gt_labels": gt_labels,
        "pred_labels": pred_labels,
        "flow": flow,
        "static_labels": static_labels,
        "gt_detections": gt_detections,
        "pred_detections": pred_detections,
        "log_info": {
            "entries": [
                {"key": "mode_default", "value": "gt"},
                {"key": "frame_file_name", "value": source_path.name},
                {"key": "frame_file_path", "value": str(source_path)},
                {"key": "visible_points", "value": int(len(points))},
                {"key": "gt_det_count", "value": len(gt_detections)},
                {"key": "eval_det_count", "value": len(pred_detections)},
                {"key": "has_flow", "value": flow is not None},
                {"key": "has_static_labels", "value": static_labels is not None},
                {"key": "demo_source", "value": "local_result_extract"},
            ]
        },
    }
