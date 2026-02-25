from src.pipeline import SkeletonPipeline

if __name__ == "__main__":

    video_path = "data/raw/marah.mp4"
    label = 0

    pipeline = SkeletonPipeline()
    pipeline.process_video(video_path, label)