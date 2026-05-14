try:
    import torch
except ImportError:
    pass
import random
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, skeleton):
        for t in self.transforms:
            skeleton = t(skeleton)
        return skeleton


class ToTensor(object):
    def __call__(self, skeleton):
        if isinstance(skeleton, np.ndarray):
            skeleton = torch.from_numpy(skeleton).float()
        return skeleton


class Jitter(object):
    """
    Apply Gaussian jitter (noise) to skeleton sequences.
    """
    def __init__(self, std_dev=0.01) -> None:
        self.std_dev = std_dev

    def __call__(self, skeleton):
        noise = np.random.normal(loc=0, scale=self.std_dev, size=skeleton.shape)
        return skeleton + noise

class Scale(object):
    """
    Apply GLOBAL spatial scaling (physically plausible).
    """
    def __init__(self, scale_range=(0.8, 1.2)) -> None:
        self.scale_range = scale_range

    def __call__(self, skeleton):
        scale = np.random.uniform(*self.scale_range)
        return skeleton * scale


class TemporalRescale(object):
    """
    Temporal rescaling using linear interpolation (smooth).
    """
    def __init__(self, temp_scaling=0.2):
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        T = len(clip)

        scale = self.L + (self.U - self.L) * np.random.random()
        new_T = max(2, int(round(T * scale)))

        # original time indices
        old_idx = np.arange(T)
        new_idx = np.linspace(0, T - 1, new_T)

        # interpolate
        resampled = []
        for i in range(clip.shape[1]):  # keypoints
            kp_traj = clip[:, i, :]  # (T, C)

            interp_kp = np.zeros((new_T, kp_traj.shape[1]))
            for c in range(kp_traj.shape[1]):
                interp_kp[:, c] = np.interp(new_idx, old_idx, kp_traj[:, c])

            resampled.append(interp_kp)

        resampled = np.stack(resampled, axis=1)  # (new_T, K, C)
        return resampled