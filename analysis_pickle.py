import pickle
import numpy as np

pkl_path = "data/pickle/pose_data_isharah2000_hands_lips_body_phase2_SI.pkl"

with open(pkl_path, "rb") as f:
    obj = pickle.load(f)

print("ROOT TYPE:", type(obj))

if isinstance(obj, dict):
    keys = list(obj.keys())
    print("JUMLAH KEY ROOT:", len(keys))
    print("5 KEY PERTAMA:", keys[:5])

    first_key = keys[0]
    sample = obj[first_key]
    print("\nSAMPLE KEY:", first_key)
    print("SAMPLE TYPE:", type(sample))

    if isinstance(sample, dict):
        print("FIELD SAMPLE:", list(sample.keys()))

        if "keypoints" in sample:
            kp = sample["keypoints"]
            print("keypoints type:", type(kp))

            if isinstance(kp, np.ndarray):
                print("keypoints shape:", kp.shape)
                print("keypoints dtype:", kp.dtype)
                print("min/max:", float(kp.min()), float(kp.max()))
            elif isinstance(kp, list):
                print("keypoints berupa list, panjang level-1:", len(kp))
                if len(kp) > 0 and isinstance(kp[0], list):
                    print("panjang level-2 (frame pertama):", len(kp[0]))
else:
    print("Objek root bukan dict, cek dengan dir(obj):", dir(obj)[:20])