"""
Standalone diagnostic tool: overlays skeleton keypoints + anatomical edges on a
single frame from a pose pkl, to sanity-check that the keypoint order matches
what this project's GCN assumes for the raw 86-point layout:
    0:21  = hand A   (Mediapipe Hands topology)
    21:42 = hand B   (Mediapipe Hands topology)
    42:61 = mouth/lips (19-point closed contour)
    61:86 = body     (Mediapipe Pose landmarks 0-24)

Edge lists below are copied read-only from modules/stgcn_layers/gcn_utils.py
(custom_hand21 / custom_mouth_8 / custom_body). Kept as plain data here (not
imported) so this script only needs numpy + matplotlib, not torch.

Usage:
    python visualize_skeleton.py --pkl datasets/pose_bisindo.pkl
    python visualize_skeleton.py --pkl datasets/pose_bisindo.pkl --key P1_S01_R1 --frame 10
"""
import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
MOUTH_EDGES = [(i, i + 1) for i in range(18)] + [(18, 0)]
BODY_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 13), (13, 15), (15, 17), (17, 19), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (18, 20), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
]

GROUPS = {
    "hand_A (0:21)": (0, 21, HAND_EDGES, "tab:red"),
    "hand_B (21:42)": (21, 42, HAND_EDGES, "tab:blue"),
    "mouth (42:61)": (42, 61, MOUTH_EDGES, "tab:green"),
    "body (61:86)": (61, 86, BODY_EDGES, "tab:orange"),
}


def pick_best_frame(kps):
    nonzero_per_frame = (~(kps == 0).all(axis=-1)).sum(axis=1)
    return int(np.argmax(nonzero_per_frame))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="datasets/pose_bisindo.pkl")
    ap.add_argument("--key", default=None, help="video_id to visualize; default: first key")
    ap.add_argument("--frame", type=int, default=None,
                     help="frame index; default: auto-pick frame with most detected points")
    ap.add_argument("--out", default="skeleton_topology_check.png")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    key = args.key or next(iter(data))
    kps = data[key]["keypoints"]  # (T, 86, 2)
    frame_idx = args.frame if args.frame is not None else pick_best_frame(kps)
    frame = kps[frame_idx]  # (86, 2)

    fig, ax = plt.subplots(figsize=(7, 9))
    for name, (start, end, edges, color) in GROUPS.items():
        pts = frame[start:end]
        valid = ~(pts == 0).all(axis=-1)
        ax.scatter(pts[valid, 0], pts[valid, 1], s=20, color=color, label=name, zorder=3)
        for i, j in edges:
            pi, pj = pts[i], pts[j]
            if (pi == 0).all() or (pj == 0).all():
                continue
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color=color, linewidth=1.3, zorder=2)
        for local_idx, p in enumerate(pts):
            if (p == 0).all():
                continue
            ax.annotate(str(local_idx), (p[0], p[1]), fontsize=5, color=color, alpha=0.85)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"key={key}  frame={frame_idx}/{len(kps)}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved to {args.out} (key={key}, frame={frame_idx})")

    # --- Quantitative cross-check: Mediapipe Pose body-wrist (idx 15/16, local to
    # body slice) should sit close to Mediapipe Hands wrist (idx 0, local to each
    # hand slice) if the 86-point order/orientation assumption is correct, since
    # both sub-systems track the same physical wrist.
    body = frame[61:86]
    hand_a = frame[0:21]
    hand_b = frame[21:42]
    print("\n--- Quantitative wrist cross-check (body Pose idx 15/16 vs Hands idx 0) ---")
    for body_wrist_idx, body_label in [(15, "body[15] (Pose left wrist)"), (16, "body[16] (Pose right wrist)")]:
        bw = body[body_wrist_idx]
        if (bw == 0).all():
            print(f"{body_label}: not detected this frame")
            continue
        for hand_name, hand in [("hand_A[0]", hand_a[0]), ("hand_B[0]", hand_b[0])]:
            if (hand == 0).all():
                continue
            dist = np.linalg.norm(bw - hand)
            print(f"{body_label} <-> {hand_name}: dist={dist:.4f}")


if __name__ == "__main__":
    main()
