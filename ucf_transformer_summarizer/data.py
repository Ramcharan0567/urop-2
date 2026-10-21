from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import Dataset

def norm_stem(s):
    return re.sub(r"[^a-z0-9]", "", Path(str(s).replace("\\", "/")).stem.lower())


def read_labels(file):
    labels = {}
    skipped = []
    anomaly_classes = (
        "abuse", "arrest", "arson", "assault", "burglary", "explosion",
        "fighting", "roadaccidents", "robbery", "shooting",
        "shoplifting", "stealing", "vandalism"
    )
    for raw in Path(file).read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        video = next((x for x in parts if ".mp4" in x.lower()), parts[0])

        if len(parts) >= 2 and parts[-1] in {"0", "1"}:
            label = int(parts[-1])
        else:
            low = line.lower().replace("\\\\", "/")
            if "normal_videos" in low or re.search(
                r"(^|[/_\\-\\s])normal([/_\\-\\s.]|$)", low
            ):
                label = 0
            elif any(cls in low for cls in anomaly_classes):
                label = 1
            else:
                skipped.append(line)
                continue

        labels[norm_stem(video)] = label

    if not labels:
        sample = "\\n".join(Path(file).read_text(errors="ignore").splitlines()[:5])
        raise ValueError(f"No labels parsed from {file}. First lines:\\n{sample}")

    print(f"Parsed {len(labels)} labels; skipped {len(skipped)} unrecognized rows.")
    if skipped:
        print("Example skipped row:", skipped[0])
    return labels


def load_feature(path):
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".npy":
        arr = np.load(p, allow_pickle=False)
    elif ext == ".npz":
        z = np.load(p, allow_pickle=False)
        arr = z["features"] if "features" in z.files else z[z.files[0]]
    elif ext in {".pt", ".pth"}:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = next(
                (obj[k] for k in ("features", "feature", "data", "arr_0") if k in obj),
                obj
            )
        arr = obj.detach().cpu().numpy() if torch.is_tensor(obj) else np.asarray(obj)
    elif ext == ".txt":
        arr = np.loadtxt(p, dtype=np.float32)
    elif ext == ".csv":
        arr = np.loadtxt(p, delimiter=",", dtype=np.float32)
    else:
        raise ValueError(f"Unsupported feature file: {p}")

    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D features, got {arr.shape}: {p}")
    if arr.shape[0] > arr.shape[1] and arr.shape[0] >= 512 and arr.shape[1] <= 512:
        arr = arr.T
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def match_features(root, labels):
    root = Path(root)
    extensions = {".txt", ".csv", ".npy", ".npz", ".pt", ".pth"}
    files = sorted(
        f for f in root.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    )

    def normalize(value):
        name = Path(str(value).replace(chr(92), "/")).stem.lower()
        if name.endswith("_c"):
            name = name[:-2]
        return re.sub(r"[^a-z0-9]", "", name)
    by_name = {}
    for f in files:
        by_name.setdefault(normalize(f.name), []).append(f)

    records = []
    unmatched = []
    for video_path, label in labels.items():
        candidates = by_name.get(normalize(video_path), [])
        if not candidates:
            unmatched.append(video_path)
            continue
        records.append((str(candidates[0]), int(label), str(video_path)))

    print(f"Discovered feature files: {len(files)}")
    print(f"Matched {len(records)} labels; unmatched: {len(unmatched)}")
    if unmatched:
        print("First unmatched:", unmatched[:5])
    return records, unmatched


class FeatureDataset(Dataset):
    def __init__(self, records, max_segments=32):
        self.records = records
        self.max_segments = max_segments

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label, *rest = self.records[index]
        features = load_feature(path)

        # Uniformly sample long sequences and zero-pad short ones.
        n = len(features)
        if n > self.max_segments:
            indices = np.linspace(
                0, n - 1, self.max_segments, dtype=int
            )
            features = features[indices]

        length = min(len(features), self.max_segments)
        result = np.zeros(
            (self.max_segments, features.shape[1]), dtype=np.float32
        )
        result[:length] = features[:length]

        return (
            torch.from_numpy(result),
            torch.tensor(int(label), dtype=torch.float32),
            torch.tensor(length, dtype=torch.long),
        )
