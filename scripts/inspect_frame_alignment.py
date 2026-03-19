#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_point_loader import load_points


PC_RANGE = (-24.0, -80.0, -10.0, 24.0, 200.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect frame alignment for PKL-indexed viewer data."
    )
    parser.add_argument("pkl_file", help="Top-level PKL file")
    parser.add_argument("--eval-dir", required=True, help="Eval directory")
    parser.add_argument("--index", type=int, default=0, help="Frame index")
    parser.add_argument("--at720", action="store_true", default=False)
    return parser.parse_args()


def load_frame_info(top_pkl: Path, index: int) -> tuple[Path, dict]:
    with open(top_pkl, "rb") as handle:
        data = pickle.load(handle)
    refs = data["infos"] if isinstance(data, dict) and "infos" in data else data
    frame_ref = refs[index]
    frame_path = Path(str(frame_ref)).expanduser()
    if not frame_path.is_absolute():
        frame_path = top_pkl.parent / frame_path
    with open(frame_path, "rb") as handle:
        info = pickle.load(handle)
    return frame_path, info


def prediction_path(eval_dir: Path, data_info: dict) -> Path:
    adrn_value = data_info.get("adrn", "")
    return eval_dir / (str(adrn_value).replace(":", "_") + ".pkl")


def point_range_mask(points: np.ndarray) -> np.ndarray:
    x_min, y_min, z_min, x_max, y_max, z_max = PC_RANGE
    return (
        (points[:, 0] > x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] > y_min)
        & (points[:, 1] < y_max)
        & (points[:, 2] > z_min)
        & (points[:, 2] < z_max)
    )


def main() -> None:
    args = parse_args()
    top_pkl = Path(args.pkl_file).expanduser()
    eval_dir = Path(args.eval_dir).expanduser()

    frame_path, info = load_frame_info(top_pkl, args.index)
    pred_path = prediction_path(eval_dir, info)

    print("frame_path =", frame_path)
    print("pred_path =", pred_path)
    print("pred_exists =", pred_path.exists())
    print("info_keys =", sorted(info.keys()))

    lidar_source = info["adrns"] if "adrns" in info else info.get("adrn")
    points, valid_mask = load_points(lidar_source, at720=args.at720)
    print("loaded_points_shape =", None if points is None else points.shape)
    print("valid_mask_len =", None if valid_mask is None else len(valid_mask))
    print("valid_mask_true =", None if valid_mask is None else int(np.sum(valid_mask)))

    if points is None or valid_mask is None:
        return

    p_range_mask = point_range_mask(points)
    print("range_true =", int(np.sum(p_range_mask)))

    gt_seg = info.get("gt_seg")
    if gt_seg is not None:
        print("gt_seg_len_raw =", len(gt_seg))
        gt_valid = np.asarray(gt_seg)[valid_mask]
        print("gt_seg_len_after_valid =", len(gt_valid))
        print("gt_seg_len_after_range =", len(gt_valid[p_range_mask]))

    flow_gt = info.get("flow_gt")
    if flow_gt is not None:
        flow_gt = np.asarray(flow_gt)
        print("flow_gt_len_raw =", len(flow_gt))
        if len(flow_gt) == len(points):
            print("flow_gt_len_after_range =", len(flow_gt[p_range_mask]))
        else:
            print("flow_gt_note = raw flow_gt length does not match loaded points")

    if not pred_path.exists():
        return

    with open(pred_path, "rb") as handle:
        pred = pickle.load(handle)

    print("counts =", pred.get("counts"))
    print("current_pts_num =", pred.get("current_pts_num"))
    for key in ["valid_pts_mask", "points_voxel_sample_mask", "pts_results", "flow_pred", "point_static"]:
        value = pred.get(key)
        print(f"{key}_len =", None if value is None else len(value))

    valid_pts_mask = pred.get("valid_pts_mask")
    if valid_pts_mask is not None:
        valid_pts_mask = np.asarray(valid_pts_mask).astype(bool)
        print("valid_pts_mask_equal_loader =", np.array_equal(valid_pts_mask, valid_mask))

    voxel_mask = pred.get("points_voxel_sample_mask")
    if voxel_mask is not None:
        voxel_mask = np.asarray(voxel_mask).astype(bool)
        print("voxel_mask_len =", len(voxel_mask))
        print("voxel_mask_true =", int(np.sum(voxel_mask)))
        if len(voxel_mask) == int(np.sum(p_range_mask)):
            print("voxel_matches_range = True")
        elif len(voxel_mask) == len(points):
            print("voxel_matches_loaded_points = True")
        else:
            print("voxel_matches_range = False")
            print("voxel_matches_loaded_points = False")


if __name__ == "__main__":
    main()
