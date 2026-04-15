"""Utilities to plot a single skeleton frame (colored by region).

Provides a small helper that reproduces the visualization from the
user-provided snippet. Works with keypoints arrays shaped (T, 86, 3)
or (T, 86, 2). Coordinates are expected to be normalized (0..1) as
produced by MediaPipe, but any 2D coordinates will be plotted as-is.

Example usage:

    from src.visualizer.plot_skeleton import plot_frame
    import numpy as np

    data = np.load("data/skeleton/00_0001.npy")  # (T,86,3)
    plot_frame(data, frame_idx=40)

Command-line usage (quick):

    python -m src.visualizer.plot_skeleton data/skeleton/00_0001.npy 40

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple


def plot_frame(keypoints: np.ndarray,
               frame_idx: int = 0,
               figsize: Tuple[float, float] = (5, 6),
               title: Optional[str] = None,
               show: bool = True,
               save_path: Optional[str] = None,
               scale_to: Optional[Tuple[float, float]] = None) -> None:
    """Plot a single frame from a keypoints array.

    Parameters
    ----------
    keypoints : np.ndarray
        Array of shape (T, 86, 3) or (T, 86, 2).
    frame_idx : int
        Frame index to plot.
    figsize : tuple
        Matplotlib figure size.
    title : Optional[str]
        Plot title. If None, a default is used.
    show : bool
        If True, call `plt.show()` at the end.
    save_path : Optional[str]
        If provided, save the figure to this path instead of or in
        addition to showing it.
    """

    if not isinstance(keypoints, np.ndarray):
        raise TypeError("keypoints must be a numpy array")

    if keypoints.ndim != 3 or keypoints.shape[1] != 86:
        raise ValueError("keypoints must have shape (T, 86, C)")

    frame = keypoints[frame_idx]

    # If input has 3 channels, use first two (x, y)
    if frame.shape[1] >= 2:
        coords = frame[:, :2].astype(float)
    else:
        raise ValueError("keypoint coordinate dimension must be >=2")

    # If requested, scale normalized coordinates (0..1) to pixel space
    # by multiplying x by width and y by height. This reproduces the
    # appearance of the second image attachment (pixel coordinates).
    if scale_to is not None:
        w, h = scale_to
        # Heuristic: only scale when coordinates look normalized (<=1.0)
        if np.nanmax(coords) <= 1.0 + 1e-6:
            coords = coords.copy()
            coords[:, 0] = coords[:, 0] * float(w)
            coords[:, 1] = coords[:, 1] * float(h)
        else:
            # coords already seem to be in pixel space; still apply
            # scaling if the caller explicitly requested it, but warn
            # the user that values >1 were detected.
            print("plot_skeleton.plot_frame: warning - coordinates look like pixel values (>1). Applying scale_to may produce incorrect results.")
            coords = coords.copy()
            coords[:, 0] = coords[:, 0] * float(w)
            coords[:, 1] = coords[:, 1] * float(h)

    pose_idx = np.arange(61, 86)
    left_hand_idx = np.arange(0, 21)
    right_hand_idx = np.arange(21, 42)
    mouth_idx = np.arange(42, 61)

    valid = ~(np.all(coords == 0, axis=1))

    def plot_part(indices: Sequence[int], color: str, label: str):
        idx = indices[valid[indices]]
        if idx.size == 0:
            return
        pts = coords[idx]
        plt.scatter(pts[:, 0], pts[:, 1], c=color, s=10, label=label)

    plt.figure(figsize=figsize)

    plot_part(pose_idx, 'blue', 'Pose')
    plot_part(left_hand_idx, 'yellow', 'Left Hand')
    plot_part(right_hand_idx, 'green', 'Right Hand')
    plot_part(mouth_idx, 'red', 'Mouth')

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    if title is None:
        title = f"Colored Skeleton Visualization - Frame {frame_idx}"
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def _main(argv):
    if len(argv) < 2:
        print("Usage: python -m src.visualizer.plot_skeleton <keypoints.npy> [frame_idx]")
        return 1

    npy_path = argv[1]
    frame_idx = int(argv[2]) if len(argv) >= 3 else 0

    data = np.load(npy_path)
    plot_frame(data, frame_idx=frame_idx)
    return 0


if __name__ == '__main__':
    raise SystemExit(_main(sys.argv))
