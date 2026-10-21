from pathlib import Path
import csv
import statistics

path = Path("transformer_test_results.csv")
if not path.exists():
    raise SystemExit(f"Cannot find {path.resolve()}")

with path.open(newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    columns = reader.fieldnames or []

print("CSV columns:", columns)
print("Total rows:", len(rows))

def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())

label_names = {
    "label", "true_label", "truelabel", "groundtruth",
    "groundtruthlabel", "actual", "target", "ytrue"
}
score_names = {
    "score", "anomalyscore", "probability", "anomalyprobability",
    "positiveprobability", "predictionscore", "prob"
}

label_col = next((c for c in columns if norm(c) in label_names), None)
score_col = next((c for c in columns if norm(c) in score_names), None)

if label_col is None or score_col is None:
    print("\nCould not safely identify the true-label and score columns.")
    print("First 3 rows:")
    for row in rows[:3]:
        print(row)
    raise SystemExit(
        "\nUse the printed column names to map the correct true-label "
        "and continuous score columns."
    )

def parse_label(value):
    v = str(value).strip().lower()
    if v in {"0", "normal", "normal video", "false"}:
        return 0
    if v in {"1", "anomaly", "abnormal", "anomalous", "true"}:
        return 1
    return None

groups = {0: [], 1: []}
bad = 0

for row in rows:
    label = parse_label(row.get(label_col, ""))
    try:
        score = float(row.get(score_col, ""))
    except (ValueError, TypeError):
        score = None

    if label is None or score is None:
        bad += 1
        continue
    groups[label].append((score, row))

print(f"Using true-label column: {label_col}")
print(f"Using score column: {score_col}")
print(f"Rows skipped: {bad}")

for label, name in [(0, "NORMAL"), (1, "ANOMALY")]:
    scores = [x[0] for x in groups[label]]
    if not scores:
        print(f"{name}: no usable rows")
        continue
    print(f"\n{name} (n={len(scores)})")
    print(f"  Mean:   {statistics.mean(scores):.6f}")
    print(f"  Median: {statistics.median(scores):.6f}")
    print(f"  Std:    {statistics.pstdev(scores):.6f}")
    print(f"  Min:    {min(scores):.6f}")
    print(f"  Max:    {max(scores):.6f}")

normal = [s for s, _ in groups[0]]
anomaly = [s for s, _ in groups[1]]

if normal and anomaly:
    print("\nSCORE OVERLAP")
    print(f"  Highest normal score:  {max(normal):.6f}")
    print(f"  Lowest anomaly score:  {min(anomaly):.6f}")
    print(f"  Anomalies scoring above the normal median: "
          f"{100 * sum(s > statistics.median(normal) for s in anomaly) / len(anomaly):.1f}%")
    print(f"  Normal videos scoring above the anomaly median: "
          f"{100 * sum(s > statistics.median(anomaly) for s in normal) / len(normal):.1f}%")

    # Rank-based ROC-AUC, calculated without requiring scikit-learn.
    combined = [(s, 0) for s in normal] + [(s, 1) for s in anomaly]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = ((i + 1) + j) / 2
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    n0, n1 = len(normal), len(anomaly)
    rank_sum_positive = sum(
        ranks[i] for i, item in enumerate(combined) if item[1] == 1
    )
    auc = (rank_sum_positive - n1 * (n1 + 1) / 2) / (n1 * n0)
    print(f"\nROC-AUC from this score column: {auc:.4f}")

    threshold = (statistics.median(normal) + statistics.median(anomaly)) / 2
    tn = sum(s <= threshold for s in normal)
    fp = len(normal) - tn
    fn = sum(s <= threshold for s in anomaly)
    tp = len(anomaly) - fn
    print(f"\nIllustrative median-midpoint threshold: {threshold:.6f}")
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("  This threshold is exploratory, not tuned or independently validated.")

    borderline = sorted(groups[1], key=lambda x: x[0])[:10]
    print("\n10 LOWEST-SCORING ANOMALOUS TEST VIDEOS")
    for score, row in borderline:
        print(f"  score={score:.6f} | " +
              str(row.get("video", row.get("path", row.get("filename", row)))))

