import importlib
import io
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "pkl_pointcloud_browser_viewer" / "app.py"


def _load_module():
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.modules.pop("pkl_pointcloud_browser_viewer.app", None)
    return importlib.import_module("pkl_pointcloud_browser_viewer.app")


def _write_pickle(path: Path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _parse_packed_frame(payload: bytes):
    point_count = int(np.frombuffer(payload[4:8], dtype=np.uint32)[0])
    name_len = int(np.frombuffer(payload[8:12], dtype=np.uint32)[0])
    offset = 16 + name_len
    offset += (-name_len) % 4
    positions = np.frombuffer(payload[offset:offset + point_count * 12], dtype=np.float32).reshape(-1, 3)
    colors = np.frombuffer(payload[offset + point_count * 12:], dtype=np.uint8).reshape(-1, 3)
    return point_count, positions, colors


def test_save_edit_rectangles_writes_json(tmp_path):
    browser = _load_module()
    output_path = tmp_path / "edit_rectangles.json"
    rectangles = {
        "frame_a.pkl": [[1.0, 2.0], [5.0, 2.0], [5.0, 9.0], [1.0, 9.0]],
    }

    result = browser.save_edit_rectangles(output_path, rectangles)

    assert result["path"] == str(output_path)
    assert result["count"] == 1
    assert output_path.exists()


def test_frame_store_aligns_point_loss_and_applies_sample_rate(tmp_path, monkeypatch):
    browser = _load_module()

    frame_info_path = tmp_path / "frame.pkl"
    index_path = tmp_path / "index.pkl"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    _write_pickle(
        frame_info_path,
        {
            "adrn": "sample_frame",
            "gt_seg": np.array([0, 0, 165, 0], dtype=np.int32),
        },
    )
    _write_pickle(index_path, {"infos": [frame_info_path.name]})
    _write_pickle(
        eval_dir / "sample_frame.pkl",
        {
            "counts": 4,
            "pts_results": np.array([1, 2, 3, 4], dtype=np.int32),
            "pts_point_loss": np.array([0.1, 0.5, 0.9, 0.7], dtype=np.float32),
        },
    )

    lidar = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        browser,
        "load_points_for_frame",
        lambda lidar_adrn, at720: (lidar, np.array([0, 1, 2, 3], dtype=np.int64)),
    )

    store = browser.FrameStore(
        pkl_file=str(index_path),
        eval_dir=str(eval_dir),
        sample_rate=2,
        label_source="pred",
        at720=False,
    )

    bundle = store._load_bundle(0)

    assert bundle["point_loss"] is not None
    assert np.allclose(bundle["point_loss"], np.array([0.1, 0.9], dtype=np.float32))
    assert np.array_equal(bundle["ignore_mask"], np.array([0, 1], dtype=np.uint8))
    assert bundle["point_loss_range"] == pytest.approx((0.1, 0.1))
    assert store.meta(fps=5.0, point_size=2.0)["hasPointLoss"] is True


def test_frame_store_load_flow_uses_gt_flow_and_mask_in_gt_mode(tmp_path, monkeypatch):
    browser = _load_module()

    frame_info_path = tmp_path / "frame.pkl"
    index_path = tmp_path / "index.pkl"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    _write_pickle(
        frame_info_path,
        {
            "adrn": "sample_frame",
            "gt_seg": np.array([0, 0, 0, 0], dtype=np.int32),
            "flow_gt": np.array(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "flow_maks": np.array([1, 0, 1, 0], dtype=np.uint8),
        },
    )
    _write_pickle(index_path, {"infos": [frame_info_path.name]})
    _write_pickle(
        eval_dir / "sample_frame.pkl",
        {
            "counts": 4,
            "pts_results": np.array([1, 2, 3, 4], dtype=np.int32),
        },
    )

    lidar = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        browser,
        "load_points_for_frame",
        lambda lidar_adrn, at720: (lidar, np.array([0, 1, 2, 3], dtype=np.int64)),
    )

    store = browser.FrameStore(
        pkl_file=str(index_path),
        eval_dir=str(eval_dir),
        sample_rate=1,
        label_source="gt",
        at720=False,
    )

    payload = store.load_flow(0, label_source="gt")

    assert payload is not None
    flow_values = np.frombuffer(payload[16:], dtype=np.float32).reshape(-1, 3)
    assert np.allclose(
        flow_values,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_frame_store_load_static_labels_uses_masked_gt_flow_for_gt_and_point_static_for_pred(tmp_path, monkeypatch):
    browser = _load_module()

    frame_info_path = tmp_path / "frame.pkl"
    index_path = tmp_path / "index.pkl"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    _write_pickle(
        frame_info_path,
        {
            "adrn": "sample_frame",
            "gt_seg": np.array([0, 0, 0, 0], dtype=np.int32),
            "flow_gt": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                ],
                dtype=np.float32,
            ),
            "flow_maks": np.array([0, 1, 0, 0], dtype=np.uint8),
        },
    )
    _write_pickle(index_path, {"infos": [frame_info_path.name]})
    _write_pickle(
        eval_dir / "sample_frame.pkl",
        {
            "counts": 4,
            "pts_results": np.array([1, 2, 3, 4], dtype=np.int32),
            "point_static": np.array([0, 1, 0, 1], dtype=np.uint8),
        },
    )

    lidar = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        browser,
        "load_points_for_frame",
        lambda lidar_adrn, at720: (lidar, np.array([0, 1, 2, 3], dtype=np.int64)),
    )

    store = browser.FrameStore(
        pkl_file=str(index_path),
        eval_dir=str(eval_dir),
        sample_rate=1,
        label_source="gt",
        at720=False,
    )

    gt_payload = store.load_static_labels(0, label_source="gt")
    pred_payload = store.load_static_labels(0, label_source="pred")

    assert gt_payload is not None
    assert pred_payload is not None
    gt_labels = np.frombuffer(gt_payload[16:], dtype=np.uint8)
    pred_labels = np.frombuffer(pred_payload[16:], dtype=np.uint8)
    assert np.array_equal(gt_labels, np.array([1, 0, 1, 1], dtype=np.uint8))
    assert np.array_equal(pred_labels, np.array([0, 1, 0, 1], dtype=np.uint8))


def test_frame_store_load_detections_includes_aligned_next_gt_boxes_only_in_gt_mode(tmp_path, monkeypatch):
    browser = _load_module()

    frame_info_path = tmp_path / "frame.pkl"
    index_path = tmp_path / "index.pkl"
    _write_pickle(
        frame_info_path,
        {
            "adrn": "sample_frame",
            "gt_boxes": np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25]], dtype=np.float32),
            "gt_names": np.array(["car"]),
            "ego2global": np.eye(4, dtype=np.float32),
            "next_labeled_frame": {
                "pose": np.array(
                    [
                        [1.0, 0.0, 0.0, 10.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                "gt": [
                    {
                        "name": "car",
                        "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                        "width": 2.0,
                        "length": 4.0,
                        "height": 1.5,
                        "theta": 0.5,
                    }
                ],
            },
        },
    )
    _write_pickle(index_path, {"infos": [frame_info_path.name]})

    lidar = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(
        browser,
        "load_points_for_frame",
        lambda lidar_adrn, at720: (lidar, np.array([0], dtype=np.int64)),
    )

    store = browser.FrameStore(
        pkl_file=str(index_path),
        eval_dir="",
        sample_rate=1,
        label_source="gt",
        at720=False,
    )

    gt_detections = store.load_detections(0, label_source="gt")
    pred_detections = store.load_detections(0, label_source="pred")

    assert len(gt_detections) == 2
    assert len(pred_detections) == 1
    assert gt_detections[1]["isNextFrame"] is True
    assert gt_detections[1]["center"] == pytest.approx([11.0, 2.0, 3.0])
    assert gt_detections[1]["size"] == pytest.approx([4.0, 2.0, 1.5])
    assert gt_detections[1]["yaw"] == pytest.approx(0.5)


def test_frame_store_load_sub_box_flows_uses_unique_gt_flow_per_split_box(tmp_path, monkeypatch):
    browser = _load_module()

    frame_info_path = tmp_path / "frame.pkl"
    index_path = tmp_path / "index.pkl"
    _write_pickle(
        frame_info_path,
        {
            "adrn": "sample_frame",
            "flow_gt": np.array(
                [
                    [0.5, 0.1, 0.0],
                    [0.5, 0.1, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.3, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "flow_maks": np.array([1, 1, 0, 0], dtype=np.uint8),
            "main_id_center": {
                7: np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                    ],
                    dtype=np.float32,
                )
            },
            "is_in_split_box": {
                7: [
                    np.array([1, 1, 0, 0], dtype=np.uint8),
                    np.array([0, 0, 1, 1], dtype=np.uint8),
                ]
            },
        },
    )
    _write_pickle(index_path, {"infos": [frame_info_path.name]})

    lidar = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(
        browser,
        "load_points_for_frame",
        lambda lidar_adrn, at720: (lidar, np.array([0], dtype=np.int64)),
    )

    store = browser.FrameStore(
        pkl_file=str(index_path),
        eval_dir="",
        sample_rate=1,
        label_source="gt",
        at720=False,
    )

    gt_items = store.load_sub_box_flows(0, label_source="gt")
    pred_items = store.load_sub_box_flows(0, label_source="pred")

    assert pred_items == []
    assert len(gt_items) == 1
    assert gt_items[0]["boxId"] == 7
    assert gt_items[0]["subBoxIndex"] == 0
    assert gt_items[0]["center"] == pytest.approx([1.0, 2.0, 3.0])
    assert gt_items[0]["flow"] == pytest.approx([0.5, 0.1, 0.0])


def test_frame_store_load_previous_points_transforms_sweeps_into_current_frame(tmp_path, monkeypatch):
    browser = _load_module()

    frame_info_path = tmp_path / "frame.pkl"
    index_path = tmp_path / "index.pkl"
    _write_pickle(
        frame_info_path,
        {
            "adrn": "sample_frame",
            "gt_seg": np.array([0], dtype=np.int32),
            "ego2global": np.eye(4, dtype=np.float32),
            "sweeps": [
                {
                    "timestamp": 1,
                    "ego2global": np.array(
                        [
                            [1.0, 0.0, 0.0, 10.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                    "adrns": ["prev_lidar"],
                }
            ],
        },
    )
    _write_pickle(index_path, {"infos": [frame_info_path.name]})

    def fake_load_points_for_frame(lidar_adrn, at720=False):
        if lidar_adrn == "sample_frame":
            return np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32), np.array([0], dtype=np.int64)
        if lidar_adrn == ["prev_lidar"]:
            return np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0]], dtype=np.float32), np.array([0], dtype=np.int64)
        raise AssertionError(f"Unexpected lidar adrn: {lidar_adrn}")

    monkeypatch.setattr(browser, "load_points_for_frame", fake_load_points_for_frame)

    store = browser.FrameStore(
        pkl_file=str(index_path),
        eval_dir="",
        sample_rate=1,
        label_source="gt",
        at720=False,
    )

    payload = store.load_previous_points(0)

    assert payload is not None
    point_count, positions, colors = _parse_packed_frame(payload)
    assert point_count == 1
    assert positions[0] == pytest.approx([11.0, 2.0, 3.0])
    assert tuple(colors[0].tolist()) == (160, 160, 160)


def test_extract_frame_image_sources_falls_back_to_matching_anno_file_entry(tmp_path):
    browser = _load_module()

    anno_file = tmp_path / "anno.pkl"
    frame_adrn = "adr::frame:prod.trip_a:lidar.mid_center_top_wide.0:lidar_ts_1"
    with open(anno_file, "wb") as handle:
        pickle.dump(
            {
                "annotation": [
                    {
                        "adrn": frame_adrn,
                        "adrns": [
                            frame_adrn,
                            "adr::frame:prod.trip_a:cam.mid_center_top_tele.0:cam_ts_1",
                            "adr::frame:prod.trip_a:cam.front_left_bottom_wide.0:cam_ts_1",
                        ],
                    },
                ]
            },
            handle,
        )

    data_info = {
        "adrn": frame_adrn,
        "adrns": [
            frame_adrn,
            "adr::frame:prod.trip_a:lidar.front_left_bottom_wide.0:lidar_ts_1",
        ],
        "anno_file": str(anno_file),
    }

    image_sources = browser.extract_frame_image_sources(data_info)

    assert image_sources == [
        {
            "cameraKey": "mid_center_top_tele",
            "cameraLabel": "mid_center_top_tele",
            "adrn": "adr::frame:prod.trip_a:cam.mid_center_top_tele.0:cam_ts_1",
        },
        {
            "cameraKey": "front_left_bottom_wide",
            "cameraLabel": "front_left_bottom_wide",
            "adrn": "adr::frame:prod.trip_a:cam.front_left_bottom_wide.0:cam_ts_1",
        },
    ]


def test_load_frame_image_encodes_normalized_browser_image(tmp_path, monkeypatch):
    browser = _load_module()

    frame_info_path = tmp_path / "frame.pkl"
    index_path = tmp_path / "index.pkl"
    frame_payload = {
        "adrn": "adr::frame:prod.trip_a:lidar.mid_center_top_wide.0:lidar_ts_1",
        "adrns": [
            "adr::frame:prod.trip_a:cam.mid_center_top_tele.0:image_ts_1",
        ],
    }
    _write_pickle(frame_info_path, frame_payload)
    _write_pickle(index_path, {"infos": [frame_info_path.name]})

    fake_utils = types.SimpleNamespace(
        bytes_to_image_array=lambda payload: np.array([[[0, 0, 255]]], dtype=np.uint8)
    )
    monkeypatch.setitem(sys.modules, "ad_cloud.adrn.data_seeker.utils", fake_utils)
    monkeypatch.setattr(browser, "_load_image_bytes_for_adrn", lambda adrn: b"fake-image")

    store = browser.FrameStore(
        pkl_file=str(index_path),
        eval_dir=None,
        sample_rate=1,
        label_source="gt",
        at720=False,
    )

    payload, content_type = store.load_frame_image(0, "mid_center_top_tele")
    image = Image.open(io.BytesIO(payload))

    assert content_type == "image/png"
    assert image.mode == "RGB"
    assert image.size == (1, 1)
    assert image.getpixel((0, 0)) == (255, 0, 0)
