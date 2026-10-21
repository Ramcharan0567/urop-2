from pathlib import Path
import re
import subprocess
import sys

from data import read_labels, match_features

archive = Path.home() / "Desktop/archive"
feature_root = archive / "Extracted C3D Features/Extracted C3D Features"
test_list = archive / "Anomaly_Test.txt"
checkpoint = Path("checkpoints/transformer_best.pt")
out_dir = Path("summaries/batch")
limit = 10

def key(value):
    name = Path(value).stem.lower()
    if name.endswith("_c"):
        name = name[:-2]
    return re.sub(r"[^a-z0-9]", "", name)

print("Loading test labels and matching features...")
labels = read_labels(test_list)
records, skipped = match_features(feature_root, labels)

print("Indexing original videos...")
video_index = {}
for p in archive.rglob("*.mp4"):
    video_index.setdefault(key(p.name), []).append(p)

out_dir.mkdir(parents=True, exist_ok=True)
success = failed = missing = 0

for i, (feature_path, label, video_key) in enumerate(records[:limit], 1):
    video_candidates = video_index.get(key(Path(feature_path).name.removesuffix("_C.txt")), [])
    if not video_candidates:
        print(f"[{i}/{limit}] VIDEO NOT FOUND: {feature_path}")
        missing += 1
        continue

    preferred = [
        v for v in video_candidates
        if ("anomaly-videos-part" in str(v).lower()) == (int(label) == 1)
    ]
    video = sorted(preferred or video_candidates, key=lambda v: len(str(v)))[0]
    output = out_dir / f"{video.stem}_summary.mp4"

    cmd = [
        sys.executable, "summarize.py",
        "--video", str(video),
        "--features", str(feature_path),
        "--checkpoint", str(checkpoint),
        "--ratio", "0.25",
        "--out", str(output),
    ]

    print(f"\n[{i}/{limit}] {video.name} (label={label})")
    try:
        subprocess.run(cmd, check=True)
        success += 1
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {video.name}: {e}")
        failed += 1

print("\n===== BATCH COMPLETE =====")
print(f"Successful summaries: {success}")
print(f"Failed: {failed}")
print(f"Videos not found: {missing}")
print(f"Output folder: {out_dir.resolve()}")
