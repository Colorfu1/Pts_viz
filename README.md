# PKL Point Cloud Browser Viewer

Standalone browser viewer for PKL-indexed point cloud datasets.

This project serves a local web app for browsing large point cloud datasets
without loading every frame into memory up front.

## Features

- Lazy per-frame loading for large datasets
- Browser playback with pause, random jump, frame-id jump, and percent jump
- GT / prediction switching
- 3D boxes, score filtering, box picking, and detail popup
- Flow and static-label overlays when present
- Resizable sidebar and log panels
- Works behind a reverse-proxy subpath as well as direct local access

## Install

### Editable install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then run:

```bash
pkl-pointcloud-viewer
```

### Script-only usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 pkl_pointcloud_viewer.py
```

## Configuration

Default config file:

`pkl_pointcloud_browser_viewer/config/viewer.yaml`

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
pkl-pointcloud-viewer /path/to/frame_index.pkl --eval-dir /path/to/eval --host 0.0.0.0
```

## Point Loading

This project intentionally does not include any SSH, private storage, or company-specific SDK logic.

### Built-in loader

The built-in loader supports frame metadata that points to local:

- `.npy`
- `.npz`
- float32 `.bin`

files, either as a single path, a list of paths, or a dict of paths.

### Custom loader

If your frame PKLs reference internal point identifiers or another storage system,
provide your own loader in one of these ways:

1. Copy `custom_point_loader.py.example` to `custom_point_loader.py`
2. Or set an external loader:

```bash
export PKL_POINTCLOUD_VIEWER_LOADER=your_package.your_module:load_points
```

The custom loader must return:

- `points`: `NxC` float array, with xyz in the first 3 columns
- `valid_mask`: boolean mask aligned with the raw point array before downstream filtering

## Repo Layout

```text
pkl_pointcloud_viewer.py
pyproject.toml
README.md
LICENSE
pkl_pointcloud_browser_viewer/
  app.py
  point_loader.py
  config/viewer.yaml
  vendor/browser_viewer/
custom_point_loader.py.example
```

## Notes

- The top-level PKL is treated as a frame index only.
- Actual points, labels, detections, flow, and static labels are loaded when a frame is requested.
- This keeps very large datasets usable in a browser workflow.
