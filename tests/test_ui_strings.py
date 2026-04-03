from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "pkl_pointcloud_browser_viewer" / "app.py"


def test_viewer_html_contains_ui_preference_persistence_hooks():
    viewer_source = MODULE_PATH.read_text(encoding="utf-8")

    assert "const VIEWER_UI_PREFERENCES_KEY =" in viewer_source
    assert "function loadUiPreferences()" in viewer_source
    assert "function saveUiPreferences()" in viewer_source
    assert "function applyStoredUiPreferences()" in viewer_source


def test_viewer_html_contains_image_panel_sidebar_controls_and_api_hooks():
    viewer_source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'data-panel-title="Image Panels"' in viewer_source
    assert 'id="imagePanelCameraSelect"' in viewer_source
    assert 'id="openImagePanel"' in viewer_source
    assert "/api/frame_images/" in viewer_source
    assert "/api/image/" in viewer_source


def test_viewer_html_contains_sub_box_flow_toggle_and_fetch_path():
    viewer_source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'id="toggleSubBoxFlow"' in viewer_source
    assert 'id="toggleSubBoxFlowText"' in viewer_source
    assert 'appUrl(`api/sub_box_flow/${index}?labelSource=${encodeURIComponent(labelSource)}`)' in viewer_source
    assert "new THREE.ArrowHelper(" in viewer_source


def test_viewer_html_contains_future_boxes_and_previous_points_toggles():
    viewer_source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'id="toggleFutureBoxes"' in viewer_source
    assert 'id="togglePrevPoints"' in viewer_source
    assert "async function fetchPrevPoints(index) {" in viewer_source
