import numpy as np


class TemporalHandler:
    def __init__(self, smoothing_window=3):
        self.window = smoothing_window

    def interpolate_missing(self, keypoints):
        """
        Interpolation for missing frames (where all keypoints are zero).
        Input: (T, 86, 3)
        """
        T = keypoints.shape[0]

        for t in range(1, T - 1):
            if np.all(keypoints[t] == 0):
                keypoints[t] = (keypoints[t - 1] + keypoints[t + 1]) / 2

        return keypoints

    def smooth(self, keypoints):
        """
        Moving average smoothing
        """
        T = keypoints.shape[0]
        smoothed = np.copy(keypoints)

        for t in range(T):
            start = max(0, t - self.window)
            end = min(T, t + self.window + 1)
            smoothed[t] = np.mean(keypoints[start:end], axis=0)

        return smoothed

    def process(self, keypoints):
        """
        Full temporal processing pipeline.
        """
        keypoints = self.interpolate_missing(keypoints)
        keypoints = self.smooth(keypoints)
        return keypoints