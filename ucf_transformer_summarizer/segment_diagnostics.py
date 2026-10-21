import csv
import json
import math
import statistics
from pathlib import Path

src = Path("transformer_test_results.csv")
dst = Path("segment_diagnostics.csv")

def logit(p):
    p = min(max(p, 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))

rows = []
with src.open(newline="") as f:
    for r in csv.DictReader(f):
        scores = json.loads(r["segment_scores"])
        scores = [float(s) for s in scores]
        n = len(scores)
        k = max(1, math.ceil(n * 0.25))

        # Reconstruct the top-k pooling operation from segment probabilities.
        top_logits = sorted((logit(s) for s in scores), reverse=True)[:k]
        reconstructed_video_score = 1 / (
            1 + math.exp(-statistics.mean(top_logits))
        )

        label = int(r["true_label"])
        pred = int(r["predicted_label"])
        top_indices = sorted(
            range(n), key=lambda i: scores[i], reverse=True
        )[:k]

        rows.append({
            "video": r["feature_path"],
            "true_label": label,
            "predicted_label": pred,
            "result": (
                "false_negative" if label == 1 and pred == 0 else
                "false_positive" if label == 0 and pred == 1 else
                "true_positive" if label == 1 else "true_negative"
            ),
            "video_score": float(r["anomaly_score"]),
            "max_segment_score": max(scores),
            "mean_segment_score": statistics.mean(scores),
            "segment_score_std": statistics.pstdev(scores),
            "segments_above_0_5": sum(s >= 0.5 for s in scores),
            "top_k": k,
            "top_k_mean_probability": statistics.mean(scores[i] for i in top_indices),
            "top_segment_indices_1based": ";".join(str(i + 1) for i in top_indices),
            "reconstructed_video_score": reconstructed_video_score,
            "reconstruction_abs_error": abs(
                reconstructed_video_score - float(r["anomaly_score"])
            )
        })

fields = list(rows[0].keys())
with dst.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", dst.resolve())
print("Videos analyzed:", len(rows))

for result in ("false_negative", "true_positive", "false_positive", "true_negative"):
    group = [r for r in rows if r["result"] == result]
    if not group:
        continue
    print(f"\n{result.upper()} (n={len(group)})")
    print(f"  Mean video score: {statistics.mean(r['video_score'] for r in group):.4f}")
    print(f"  Mean max segment score: {statistics.mean(r['max_segment_score'] for r in group):.4f}")
    print(f"  Mean segments >= 0.5: {statistics.mean(r['segments_above_0_5'] for r in group):.2f}")
    print("  Lowest video scores:")
    for r in sorted(group, key=lambda x: x["video_score"])[:5]:
        print(f"    {r['video']}: video={r['video_score']:.4f}, "
              f"max_segment={r['max_segment_score']:.4f}")

print("\nLargest pooling reconstruction error:",
      f"{max(r['reconstruction_abs_error'] for r in rows):.8f}")
