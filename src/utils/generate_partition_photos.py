import cv2
import numpy as np
import os
import sys

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.extractor.holistic_86 import Holistic86Extractor
from src.visualizer.draw_skeleton import SkeletonDrawer

def main():
    video_path = r"C:\TA\Source-Code\rgb-to-skeleton-mediapipe\data\raw\AKU CIUM BAU BADAN DIA\ACHMAD_AKU CIUM BAU BADAN DIA_05.mp4"
    if not os.path.exists(video_path):
        print("Video not found at path:", video_path)
        # Try to find any mp4 in the data directory
        return

    # Extract one frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 91) # Frame 167 usually has action
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read frame.")
        return

    cv2.imwrite("original_photo.jpg", frame)

    extractor = Holistic86Extractor()
    keypoints = extractor.extract_frame(frame)

    # Partition ranges based on keypoint selector
    # GL: 0-20 (21 points) -> 0:21
    # GR: 21-41 (21 points) -> 21:42
    # GM: 42-60 (19 points) -> 42:61
    # GP: 61-85 (25 points) -> 61:86
    ranges = {
        "GL": (0, 21),
        "GR": (21, 42),
        "GM": (42, 61),
        "GP": (61, 86)
    }

    h, w, _ = frame.shape
    drawer = SkeletonDrawer(resolution=(w, h))

    for name, (start, end) in ranges.items():
        # Mask out other keypoints by setting them to zero
        kp_masked = np.zeros_like(keypoints)
        kp_masked[start:end] = keypoints[start:end]
        
        # Draw purely the skeleton without RGB overlay
        rendered = drawer.draw_frame(kp_masked, background=None)
        
        output_filename = f"{name}_partition.jpg"
        cv2.imwrite(output_filename, rendered)
        print(f"Generated {output_filename}")

if __name__ == "__main__":
    main()
