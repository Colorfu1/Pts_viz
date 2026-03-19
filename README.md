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

## Run

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

## Configuration

Base config file:

`pkl_pointcloud_browser_viewer/config/viewer.yaml`

Optional local override:

`pkl_pointcloud_browser_viewer/config/viewer.local.yaml`

The local override is loaded automatically when it exists and is intended for
machine-specific paths that should not be committed. Start from:

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
pkl-pointcloud-viewer /path/to/frame_index.pkl --eval-dir /path/to/eval --host 0.0.0.0
```

Startup prints the effective config files and whether `at720` is enabled.

## Point Loading

This project does not include any SSH or tunnel logic. It serves a browser app over plain HTTP.

## Data Flow

The viewer does not pass the top-level frame-index PKL directly into a loader.
The actual flow is:

1. `python3 pkl_pointcloud_viewer.py /path/to/frame_index.pkl`
2. `pkl_pointcloud_browser_viewer/app.py` parses `pkl_file`
3. `FrameStore._load_frame_index()` loads the top-level PKL and expands `l2` / `l3` / `infos`
4. those entries become `frame_paths`
5. when the browser requests frame `N`, `FrameStore._load_bundle(N)` takes `frame_paths[N]` as the per-frame `source_file`
6. `pkl_pointcloud_browser_viewer/point_loader.py` dispatches that `source_file` to the first loader that can handle it
7. the loader returns one frame bundle dict with points / labels / flow / static / detections / log info
8. `app.py` normalizes that bundle and serves `/api/frame`, `/api/det`, `/api/flow`, `/api/static`, and `/api/frame_info`

So:

- online/internal datasets usually have per-frame `.pkl` entries, which are handled by `custom_point_loader.py`
- repo demo datasets use per-frame `.npz` entries, which are handled by `pkl_pointcloud_browser_viewer/demo_point_loader.py`

Key code locations:

- `pkl_pointcloud_viewer.py`
- `pkl_pointcloud_browser_viewer/app.py`
- `pkl_pointcloud_browser_viewer/point_loader.py`
- `custom_point_loader.py`
- `pkl_pointcloud_browser_viewer/demo_point_loader.py`

### Built-in loader

The built-in loader supports frame metadata that points to local:

- `.npy`
- `.npz`
- float32 `.bin`

files, either as a single path, a list of paths, or a dict of paths.

### Frame bundle loader

The viewer now prefers a frame-level loader interface:

```python
load_frame_bundle(source_file, *, eval_dir=None, at720=False) -> dict | None
```

`source_file` is the per-frame source entry from the top-level frame index.
The returned dict can contain everything the viewer needs to render one frame:

- `points` or `positions`
- `gt_labels`
- `pred_labels`
- `flow`
- `static_labels`
- `gt_detections`
- `pred_detections`
- `log_info`

If your dataset references internal point identifiers or another storage system,
provide your own loader in one of these ways:

1. Copy `custom_point_loader.py.example` to `custom_point_loader.py`
2. Or set an external loader module:

```bash
export PKL_POINTCLOUD_VIEWER_LOADER=your_package.your_module:load_frame_bundle
```

The repository currently ships with two concrete loaders:

- [custom_point_loader.py](custom_point_loader.py)
  Xiaomi-internal ADRN loader for online PKL datasets
- [pkl_pointcloud_browser_viewer/demo_point_loader.py](pkl_pointcloud_browser_viewer/demo_point_loader.py)
  repo-local demo loader for bundled sample data

`load_points(...)` is still supported as a compatibility helper for diagnostics such as
`scripts/inspect_frame_alignment.py`.

## AT720

`at720` can be enabled in config or by CLI:

```bash
pkl-pointcloud-viewer --at720
```

For internal datasets, the viewer also auto-enables `at720` when it detects the marker from
frame index entries or the first frame metadata. Use `--no-at720` to disable that behavior.

## Demo Data

Minimal sample data is included under:

`examples/demo_dataset/`

Files:

- `examples/demo_dataset/demo_frame_index.pkl`
- `examples/demo_dataset/frame_000.npz`
- `examples/demo_dataset/frame_001.npz`

Run the minimal demo:

```bash
python3 pkl_pointcloud_viewer.py examples/demo_dataset/demo_frame_index.pkl
```

Then open:

`http://127.0.0.1:8766/`

To bind for remote access:

```bash
python3 pkl_pointcloud_viewer.py examples/demo_dataset/demo_frame_index.pkl --host 0.0.0.0
```

The bundled demo loader reads those `.npz` files and returns one full frame bundle with
points, GT/pred labels, flow, and OD boxes.

The current demo files are extracted from local real result files under
`/home/mi/data/data_pkl/trt_out/ADLSVC-156226` with a fixed stride to keep the repo small.
Use `scripts/build_demo_from_local_results.py` to regenerate them from another local result folder.

Example:

```bash
python3 scripts/build_demo_from_local_results.py \
  /home/mi/data/data_pkl/trt_out/ADLSVC-156226 \
  --output-dir examples/demo_dataset \
  --frames 2 \
  --stride 16
```

Stop the viewer with `Ctrl+C`.

## Debugging

The repository includes a small alignment inspector:

```bash
python3 scripts/inspect_frame_alignment.py /path/to/frame_index.pkl --eval-dir /path/to/eval --index 0 --at720
```

Use it to compare point counts, masks, and prediction lengths for one frame when dataset alignment fails.

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
custom_point_loader.py
scripts/inspect_frame_alignment.py
scripts/build_demo_from_local_results.py
examples/demo_dataset/
```

## Notes

- The top-level PKL is treated as a frame index only.
- Actual points, labels, detections, flow, and static labels are loaded when a frame is requested.
- This keeps very large datasets usable in a browser workflow.
