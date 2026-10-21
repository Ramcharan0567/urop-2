from pathlib import Path
import csv
import json
import statistics

root = Path.home() / "Desktop/ucf_transformer_summarizer/summaries/mixed_batch"
manifest = root / "batch_manifest.csv"
output = root / "score_analysis.csv"

def get_scores(report):
    data = json.loads(report.read_text())
    scores = data.get("all_segment_scores", data.get("scores", []))
    vals = []
    for i, item in enumerate(scores):
        if isinstance(item, (int, float)):
            vals.append(float(item))
        elif isinstance(item, dict):
            score = item.get("score", item.get("anomaly_score", item.get("importance")))
            if score is not None:
                vals.append(float(score))
    return vals

rows = []
with manifest.open(newline="") as f:
    for item in csv.DictReader(f):
        report_path = Path(item["report_path"]) if item["report_path"] else None
        scores = get_scores(report_path) if report_path and report_path.exists() else []
        if not scores:
            print(f"No scores found for {item['video']}; inspect its JSON report.")
            continue
        rows.append({
            "video": item["video"],
            "class": item["class"],
            "label": item["label"],
            "segment_count": len(scores),
            "mean_score": statistics.mean(scores),
            "std_score": statistics.pstdev(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_range": max(scores) - min(scores),
            "top_segment_index": max(range(len(scores)), key=scores.__getitem__),
            "top_segment_score": max(scores)
        })

if rows:
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for cls in ("normal", "anomaly"):
        group = [r for r in rows if r["class"] == cls]
        if group:
            print(
                f"{cls.upper():7} n={len(group)} | "
                f"mean of video mean-scores="
                f"{statistics.mean(r['mean_score'] for r in group):.6f} | "
                f"mean score range="
                f"{statistics.mean(r['score_range'] for r in group):.6f}"
            )
    print(f"\nCSV saved: {output}")
else:
    print("No score rows generated. Check the JSON reports and manifest.")
