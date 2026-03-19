# PKL Point Cloud Browser Viewer

Browser viewer for large PKL-indexed point cloud datasets.

The top-level PKL is treated as a frame index. Actual points, labels, flow,
static labels, and detections are loaded only when the browser requests a
frame.

## Features

- Lazy per-frame loading
- Playback, pause, random jump, frame-id jump, and percent jump
- GT / prediction switching
- 3D boxes, score filtering, picking, and detail popup
- Flow and static-label overlays when present
- Resizable sidebar and log panels
- Direct local access or reverse-proxy subpath access

## Quick Start

Install dependencies and run:

```bash
pip install -r requirements.txt
python3 pkl_pointcloud_viewer.py
```

Open:

`http://127.0.0.1:8766/`

For remote access:

```bash
python3 pkl_pointcloud_viewer.py --host 0.0.0.0
```

## Demo

Bundled sample data lives in:

`examples/demo_dataset/`

Run the demo:

```bash
python3 pkl_pointcloud_viewer.py examples/demo_dataset/demo_frame_index.pkl
```

The bundled demo loader reads the per-frame `.npz` files in that directory and
returns a full frame bundle with points, labels, flow, and OD boxes.

To regenerate the demo from another local result folder:

```bash
python3 scripts/build_demo_from_local_results.py \
  /path/to/local_result_dir \
  --output-dir examples/demo_dataset \
  --frames 2 \
  --stride 16
```

Stop the viewer with `Ctrl+C`.

## Configuration

Base config:

`pkl_pointcloud_browser_viewer/config/viewer.yaml`

Optional local override:

`pkl_pointcloud_browser_viewer/config/viewer.local.yaml`

Start from:

`pkl_pointcloud_browser_viewer/config/viewer.local.yaml.example`

Example:

```yaml
pkl_file: /path/to/frame_index.pkl
eval_dir: /path/to/eval
host: 127.0.0.1
port: 8766
fps: 5.0
point_size: 2.0
sample_rate: 1
label_source: gt
at720: false
open_browser: false
```

CLI arguments override YAML:

```bash
python3 pkl_pointcloud_viewer.py /path/to/frame_index.pkl --eval-dir /path/to/eval --host 0.0.0.0
```

## Custom Loader

The viewer resolves a top-level frame index into per-frame `source_file`
entries, then dispatches each `source_file` to the first loader that can handle
it.

Preferred loader interface:

```python
load_frame_bundle(source_file, *, eval_dir=None, at720=False) -> dict | None
```

Return a dict containing any of:

- `points` or `positions`
- `gt_labels`
- `pred_labels`
- `flow`
- `static_labels`
- `gt_detections`
- `pred_detections`
- `log_info`

Two ways to add your own loader:

1. Copy `custom_point_loader.py.example` to `custom_point_loader.py`
2. Set `PKL_POINTCLOUD_VIEWER_LOADER=your_package.your_module:load_frame_bundle`

Built-in support covers frame metadata that points to local `.npy`, `.npz`, or
float32 `.bin` files. The repo also ships with
[demo_point_loader.py](pkl_pointcloud_browser_viewer/demo_point_loader.py) for
the bundled demo dataset.

## How It Works

1. `pkl_pointcloud_viewer.py` parses `pkl_file`
2. `FrameStore._load_frame_index()` expands the top-level PKL into `frame_paths`
3. When frame `N` is requested, `frame_paths[N]` becomes the per-frame `source_file`
4. `point_loader.py` dispatches that `source_file` to a loader
5. The returned frame bundle is normalized and served through `/api/frame`, `/api/det`, `/api/flow`, `/api/static`, and `/api/frame_info`

For the bundled demo, the per-frame `source_file` entries are `.npz` files. For
other datasets, provide your own loader.

## Repo Layout

```text
pkl_pointcloud_viewer.py
pyproject.toml
README.md
LICENSE
pkl_pointcloud_browser_viewer/
  app.py
  demo_point_loader.py
  point_loader.py
  config/viewer.yaml
  config/viewer.local.yaml.example
custom_point_loader.py.example
scripts/build_demo_from_local_results.py
examples/demo_dataset/
```
