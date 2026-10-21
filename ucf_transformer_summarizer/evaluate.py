import argparse, csv, json
import numpy as np, torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from data import read_labels, match_features, FeatureDataset
from model import TemporalTransformerSummarizer

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--feature-root",required=True); p.add_argument("--test-list",required=True)
    p.add_argument("--checkpoint",default="checkpoints/transformer_best.pt")
    p.add_argument("--out",default="transformer_test_scores.csv"); a=p.parse_args()
    dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    c=torch.load(a.checkpoint,map_location="cpu",weights_only=False); ms=c["config"]["max_segments"]
    m=TemporalTransformerSummarizer(max_segments=ms); m.load_state_dict(c["model_state"]); m.to(dev).eval()
    rec,skipped=match_features(a.feature_root,read_labels(a.test_list)); print(f"Matched {len(rec)} test features; unmatched files: {skipped}")
    if not rec: raise RuntimeError("No test features matched.")
    ds=FeatureDataset(rec,ms); ys=[]; ps=[]; rows=[]
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, n = ds[i]
            path = rec[i][2]
            z, segment_logits = m(x[None].to(dev), n[None].to(dev))
            prob = torch.sigmoid(z)[0].item()
            scores = torch.sigmoid(segment_logits[0, :int(n)]).cpu().numpy().tolist()
            ys.append(int(y))
            ps.append(prob)
            rows.append([path, int(y), int(prob >= 0.5), prob, json.dumps(scores)])
    pred=(np.array(ps)>=.5).astype(int)
    print("Confusion matrix [[TN,FP],[FN,TP]]:\n",confusion_matrix(ys,pred,labels=[0,1]))
    print(f"Accuracy={accuracy_score(ys,pred):.4f}; ROC-AUC={roc_auc_score(ys,ps):.4f}")
    pr,re,f1,_=precision_recall_fscore_support(ys,pred,average="binary",zero_division=0)
    print(f"Anomaly precision={pr:.4f}; recall={re:.4f}; F1={f1:.4f}")
    with open(a.out,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["feature_path","true_label","predicted_label","anomaly_score","segment_scores"]); w.writerows(rows)
    print("Saved",a.out)
if __name__=="__main__": main()
