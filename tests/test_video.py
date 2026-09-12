import numpy as np

from ofi import make_pair, make_texture, translation_flow
from ofi.video import WindowTracker, crop_letterbox

SHAPE = (96, 128)


def test_crop_letterbox_removes_black_bands():
    frames = np.zeros((3, *SHAPE))
    frames[:, 20:76, :] = 0.5
    assert crop_letterbox(frames).shape == (3, 56, 128)


def test_tracker_lands_on_subject_not_camera_motion():
    """Whole frame pans by (2, 0); a blob at (90, 30) additionally moves (0, 3)."""
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]].astype(float)
    u = np.full(SHAPE, 2.0)
    v = 3.0 * np.exp(-((xs - 90) ** 2 + (ys - 30) ** 2) / (2 * 8.0**2))
    tracker = WindowTracker(SHAPE, width=10.0)
    f, center = tracker.window(u, v)
    assert f.shape == SHAPE
    assert np.hypot(center[0] - 90, center[1] - 30) < 6
    # follows the blob: after a step the centre has moved with the flow (+2, +3)
    _, center2 = tracker.window(u, v)
    assert center2[0] > center[0] and center2[1] > center[1]
