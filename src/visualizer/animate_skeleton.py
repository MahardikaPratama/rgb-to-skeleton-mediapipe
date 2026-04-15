import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from src.visualizer.plot_skeleton import plot_frame

def create_skeleton_animation_gif(keypoints_data: np.ndarray, output_path: str, interval=40):
    """
    Generates a GIF animation from skeleton keypoints.

    Args:
        keypoints_data (np.ndarray): Skeleton data of shape (T, 86, 3).
        output_path (str): Path to save the output GIF file.
        interval (int): Delay between frames in milliseconds. 
                        40ms corresponds to 25 FPS.
    """
    
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.close(fig) # Prevent static figure from displaying

    # Set axis limits based on normalized coordinates
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0) # Invert Y-axis to match image coordinates
    ax.set_aspect('equal')
    ax.axis('off') # Hide axes

    # The update function for the animation
    def update(frame_idx):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect('equal')
        ax.axis('off')
        
        frame_kps = keypoints_data[frame_idx]
        
        # Use the existing plotting function but draw on the animation's axis
        # Pass the full keypoints_data array and the current frame_idx
        plot_frame(keypoints_data, frame_idx=frame_idx, ax=ax, show=False)
        
        ax.set_title(f"Frame: {frame_idx + 1}/{len(keypoints_data)}", fontsize=10)

    # Create the animation
    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(keypoints_data),
        interval=interval
    )

    # Save as GIF
    # This can be slow. The progress_callback shows progress in the console.
    progress_callback = lambda i, n: print(f'Saving frame {i+1}/{n}')
    ani.save(output_path, writer='pillow', fps=1000/interval, progress_callback=progress_callback)
    
    return output_path
