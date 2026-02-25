"""
Test Script untuk memeriksa perbaikan face keypoints
Menguji smooth dan presisi pada area wajah
Author: Mahardika  
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import SkeletonPipeline
from src.config import DATA_DIR, RAW_VIDEO_DIR
from src.visualizer.preview_generator import PreviewGenerator


def test_face_smoothness(video_path, output_name):
    """
    Test smoothness dari face keypoints sebelum dan sesudah enhancement
    """
    print(f"Testing face smoothness untuk: {video_path}")
    
    # Proses dengan pipeline yang sudah diperbaiki
    pipeline = SkeletonPipeline()
    keypoints = pipeline.process_video(video_path, output_name)
    
    # Analisis smoothness khusus untuk face area (index 54-86)
    face_keypoints = keypoints[:, 54:86, :]
    
    # Hitung variance movement untuk setiap face keypoint
    face_smoothness_scores = []
    
    for kp_idx in range(face_keypoints.shape[1]):
        for coord_idx in range(face_keypoints.shape[2]):
            signal = face_keypoints[:, kp_idx, coord_idx]
            
            # Skip jika semua nol
            if np.all(signal == 0):
                continue
                
            # Hitung derivative (kecepatan perubahan)
            velocity = np.diff(signal)
            acceleration = np.diff(velocity)
            
            # Smoothness score = 1/variance dari acceleration (semakin tinggi = semakin smooth)
            if len(acceleration) > 0 and np.var(acceleration) > 0:
                smoothness = 1.0 / (np.var(acceleration) + 1e-8)
                face_smoothness_scores.append(smoothness)
    
    avg_smoothness = np.mean(face_smoothness_scores) if face_smoothness_scores else 0
    
    print(f"Face Smoothness Score: {avg_smoothness:.4f}")
    print(f"Total face keypoints analyzed: {len(face_smoothness_scores)}")
    
    return keypoints, avg_smoothness


def analyze_face_precision(keypoints):
    """
    Analisis presisi face keypoints
    """
    face_keypoints = keypoints[:, 54:86, :]
    
    # Hitung stabilitas untuk area mata (index 10-21 dalam face keypoints)
    eye_keypoints = face_keypoints[:, 10:22, :]
    eye_stability = []
    
    for kp_idx in range(eye_keypoints.shape[1]):
        for coord_idx in range(eye_keypoints.shape[2]):
            signal = eye_keypoints[:, kp_idx, coord_idx] 
            
            if np.all(signal == 0):
                continue
                
            # Hitung standard deviation (semakin kecil = semakin stabil)
            stability = 1.0 / (np.std(signal) + 1e-8)
            eye_stability.append(stability)
    
    avg_eye_stability = np.mean(eye_stability) if eye_stability else 0
    
    # Hitung stabilitas untuk area mulut (index 25-31 dalam face keypoints)
    mouth_keypoints = face_keypoints[:, 25:32, :]
    mouth_stability = []
    
    for kp_idx in range(mouth_keypoints.shape[1]):
        for coord_idx in range(mouth_keypoints.shape[2]):
            signal = mouth_keypoints[:, kp_idx, coord_idx]
            
            if np.all(signal == 0):
                continue
                
            stability = 1.0 / (np.std(signal) + 1e-8) 
            mouth_stability.append(stability)
    
    avg_mouth_stability = np.mean(mouth_stability) if mouth_stability else 0
    
    print(f"Eye Area Stability: {avg_eye_stability:.4f}")
    print(f"Mouth Area Stability: {avg_mouth_stability:.4f}")
    
    return avg_eye_stability, avg_mouth_stability


def plot_face_keypoint_trajectory(keypoints, save_path="face_trajectory_analysis.png"):
    """
    Plot trajectory beberapa key face keypoints untuk visual inspection
    """
    face_keypoints = keypoints[:, 54:86, :]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Face Keypoints Trajectory Analysis', fontsize=14)
    
    # Plot eye keypoints (ambil beberapa sampel)
    axes[0, 0].plot(face_keypoints[:, 10, 0], label='Left Eye Corner X')  
    axes[0, 0].plot(face_keypoints[:, 16, 0], label='Right Eye Corner X')
    axes[0, 0].set_title('Eye X Coordinates')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(face_keypoints[:, 10, 1], label='Left Eye Corner Y')
    axes[0, 1].plot(face_keypoints[:, 16, 1], label='Right Eye Corner Y') 
    axes[0, 1].set_title('Eye Y Coordinates')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot mouth keypoints
    axes[1, 0].plot(face_keypoints[:, 25, 0], label='Mouth Left X')
    axes[1, 0].plot(face_keypoints[:, 28, 0], label='Mouth Right X') 
    axes[1, 0].set_title('Mouth X Coordinates')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(face_keypoints[:, 25, 1], label='Mouth Left Y')
    axes[1, 1].plot(face_keypoints[:, 28, 1], label='Mouth Right Y')
    axes[1, 1].set_title('Mouth Y Coordinates') 
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Trajectory plot saved to: {save_path}")


def main():
    """
    Main test function
    """
    # Cari file video untuk testing
    video_files = []
    for file in os.listdir(RAW_VIDEO_DIR):
        if file.endswith(('.mp4', '.avi', '.mov')):
            video_files.append(os.path.join(RAW_VIDEO_DIR, file))
    
    if not video_files:
        print("Tidak ada file video ditemukan di folder raw/")
        print(f"Pastikan ada file video di: {RAW_VIDEO_DIR}")
        return
    
    # Test dengan file video pertama yang ditemukan
    video_path = video_files[0]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    print("="*60)
    print("FACE KEYPOINTS ENHANCEMENT TEST")
    print("="*60)
    print(f"Testing dengan video: {video_name}")
    print()
    
    # Test smoothness dan precision
    keypoints, smoothness_score = test_face_smoothness(video_path, video_name)
    
    print()
    print("Analyzing face precision...")
    eye_stability, mouth_stability = analyze_face_precision(keypoints)
    
    print()
    print("="*60)
    print("HASIL TEST:")
    print(f"- Face Smoothness Score: {smoothness_score:.4f}")
    print(f"- Eye Area Stability: {eye_stability:.4f}")  
    print(f"- Mouth Area Stability: {mouth_stability:.4f}")
    print("="*60)
    
    # Generate trajectory plot
    print()
    print("Generating trajectory analysis plot...")
    plot_face_keypoint_trajectory(keypoints)
    
    print()
    print("Test completed! Cek hasil:")
    print(f"1. Skeleton output: {os.path.join(DATA_DIR, 'skeleton', video_name + '.npy')}")
    print(f"2. Preview video: {os.path.join(DATA_DIR, 'preview')}")
    print("3. Trajectory plot: face_trajectory_analysis.png")


if __name__ == "__main__":
    main()