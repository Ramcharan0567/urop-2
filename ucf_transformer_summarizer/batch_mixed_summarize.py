from pathlib import Path
import subprocess, sys, csv, json, re

BASE = Path.home() / "Desktop"
ARCHIVE = BASE / "archive"
PROJECT = BASE / "ucf_transformer_summarizer"
FEATURE_ROOT = ARCHIVE / "Extracted C3D Features" / "Extracted C3D Features"
TEST_LIST = ARCHIVE / "Anomaly_Test.txt"
OUT = PROJECT / "summaries" / "mixed_batch"

def key(s):
    s = Path(s).stem.lower()
    s = re.sub(r"_c$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)

videos = {}
for p in ARCHIVE.rglob("*.mp4"):
    videos.setdefault(key(p.name), p)

features = {}
for p in FEATURE_ROOT.rglob("*.txt"):
    features.setdefault(key(p.name), p)

groups = {0: [], 1: []}
for line in TEST_LIST.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    label = 0 if "normal" in line.lower() else 1
    k = key(line)
    if k in videos and k in features:
        groups[label].append((line, videos[k], features[k]))

for label in groups:
    # Deduplicate by video key
    unique = {}
    for entry in groups[label]:
        unique[key(entry[0])] = entry
    groups[label] = list(unique.values())

if len(groups[0]) < 5 or len(groups[1]) < 5:
    sys.exit(f"Not enough matched files: normal={len(groups[0])}, anomaly={len(groups[1])}")

selected = [(0, x) for x in groups[0][:5]] + [(1, x) for x in groups[1][:5]]
OUT.mkdir(parents=True, exist_ok=True)
manifest = []

for i, (label, (entry, video, feature)) in enumerate(selected, 1):
    output = OUT / f"{i:02d}_{video.stem}_summary.mp4"
    print(f"\n[{i}/10] {'NORMAL' if label == 0 else 'ANOMALY'}: {video.name}", flush=True)
    cmd = [
        sys.executable, str(PROJECT / "summarize.py"),
        "--video", str(video),
        "--features", str(feature),
        "--checkpoint", str(PROJECT / "checkpoints" / "transformer_best.pt"),
        "--ratio", "0.25",
        "--out", str(output)
    ]
    result = subprocess.run(cmd, cwd=PROJECT)
    report = output.with_suffix(".segments.json")
    if result.returncode == 0 and report.exists():
        manifest.append({
            "video": video.name,
            "label": label,
            "class": "normal" if label == 0 else "anomaly",
            "video_path": str(video),
            "feature_path": str(feature),
            "summary_path": str(output),
            "report_path": str(report)
        })
    else:
        print(f"[FAILED] {video.name}")

with (OUT / "batch_manifest.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "video", "label", "class", "video_path",
        "feature_path", "summary_path", "report_path"
    ])
    writer.writeheader()
    writer.writerows(manifest)

print(f"\nSuccessful summaries: {len(manifest)}/10")
print(f"Manifest: {OUT / 'batch_manifest.csv'}")
