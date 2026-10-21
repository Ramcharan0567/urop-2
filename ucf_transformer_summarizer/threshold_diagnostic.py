import csv
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

with open("transformer_test_results.csv", newline="") as f:
    rows = list(csv.DictReader(f))

y = np.array([int(r["true_label"]) for r in rows])
scores = np.array([float(r["anomaly_score"]) for r in rows])

print("Threshold | Accuracy | Precision | Recall | F1 | FP | FN")
print("-" * 62)

for t in [0.10, 0.20, 0.30, 0.40, 0.439456, 0.50, 0.60, 0.70, 0.80]:
    p = (scores >= t).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())

    print(
        f"{t:9.3f} | {accuracy_score(y,p):8.3f} | "
        f"{precision_score(y,p,zero_division=0):9.3f} | "
        f"{recall_score(y,p,zero_division=0):6.3f} | "
        f"{f1_score(y,p,zero_division=0):.3f} | {fp:2d} | {fn:2d}"
    )
