#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


BBOX_CLASS_TO_SEG_CLASS = {
    0: 4,
    1: 6,
    2: 5,
    3: 7,
    4: 8,
    5: 12,
    6: 12,
}

BBOX_CLASS_TO_NAME = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "pedestrian",
    4: "cyclist",
    5: "barrier",
    6: "barrier",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build repo demo dataset from local *_pc/_seg/_det/_flow result files."
    )
    parser.add_argument("source_dir", help="Directory containing *_pc.npy and related files")
    parser.add_argument(
        "--output-dir",
        default="examples/demo_dataset",
        help="Output directory inside the repo",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=2,
        help="Number of frames to export",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Keep every Nth visible point to keep the demo small",
    )
    return parser.parse_args()


def stem_from_pc(path: Path) -> str:
    return path.name.split("_pc.npy")[0]


def convert_frame(frame_path: Path, output_path: Path, stride: int) -> None:
    stem = stem_from_pc(frame_path)
    root = frame_path.parent
    seg_path = root / f"{stem}_seg.npy"
    det_path = root / f"{stem}_det.npy"
    flow_path = root / f"{stem}_flow.npy"

    points = np.load(frame_path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] < 5:
        raise ValueError(f"Unexpected point shape for {frame_path}: {points.shape}")
    visible_mask = points[:, 4] == 0
    visible_points = points[visible_mask]

    seg = np.load(seg_path, allow_pickle=False).reshape(-1)
    flow = np.load(flow_path, allow_pickle=False)
    if len(seg) != len(visible_points):
        raise ValueError(f"Seg mismatch for {stem}: {len(seg)} vs {len(visible_points)}")
    if len(flow) != len(visible_points):
        raise ValueError(f"Flow mismatch for {stem}: {len(flow)} vs {len(visible_points)}")

    visible_points = visible_points[::stride]
    seg = seg[::stride].astype(np.int32, copy=False)
    flow = flow[::stride].astype(np.float32, copy=False)

    det = np.load(det_path, allow_pickle=True)
    if det.ndim == 1:
        det = np.expand_dims(det, axis=0)
    if det.size == 0:
        pred_boxes = np.empty((0, 7), dtype=np.float32)
        pred_bbox_classes = np.empty((0,), dtype=np.int32)
        pred_scores = np.empty((0,), dtype=np.float32)
        pred_seg_classes = np.empty((0,), dtype=np.int32)
        pred_names: list[str] = []
    else:
        pred_boxes = det[:, :7].astype(np.float32, copy=False)
        pred_bbox_classes = det[:, -1].round().astype(np.int32, copy=False)
        pred_scores = det[:, -2].astype(np.float32, copy=False)
        pred_seg_classes = np.array(
            [BBOX_CLASS_TO_SEG_CLASS.get(int(x), 12) for x in pred_bbox_classes],
            dtype=np.int32,
        )
        pred_names = [BBOX_CLASS_TO_NAME.get(int(x), f"class_{int(x)}") for x in pred_bbox_classes]

    np.savez_compressed(
        output_path,
        points=visible_points.astype(np.float32, copy=False),
        gt_labels=seg,
        pred_labels=seg,
        flow=flow,
        gt_boxes=pred_boxes,
        gt_bbox_classes=pred_bbox_classes,
        gt_seg_classes=pred_seg_classes,
        gt_names=np.array(pred_names, dtype="<U32"),
        pred_boxes=pred_boxes,
        pred_bbox_classes=pred_bbox_classes,
        pred_seg_classes=pred_seg_classes,
        pred_names=np.array(pred_names, dtype="<U32"),
        pred_scores=pred_scores,
    )


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(source_dir.glob("*_pc.npy"))
    if not frame_paths:
        raise FileNotFoundError(f"No *_pc.npy files found in {source_dir}")

    selected = frame_paths[: args.frames]
    exported: list[str] = []
    for idx, frame_path in enumerate(selected):
        out_path = output_dir / f"frame_{idx:03d}.npz"
        convert_frame(frame_path, out_path, stride=max(1, args.stride))
        exported.append(out_path.name)

    index_path = output_dir / "demo_frame_index.pkl"
    with open(index_path, "wb") as handle:
        pickle.dump({"infos": exported}, handle)

    print(f"Exported {len(exported)} frames to {output_dir}")
    for name in exported:
        print(f"  {name}")
    print(index_path)


if __name__ == "__main__":
    main()
