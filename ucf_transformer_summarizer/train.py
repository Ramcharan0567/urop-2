import argparse, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from data import read_labels, match_features, FeatureDataset
from model import TemporalTransformerSummarizer

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--feature-root", required=True); p.add_argument("--train-list", required=True)
    p.add_argument("--out", default="checkpoints/transformer_best.pt")
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--max-segments", type=int, default=128)
    a=p.parse_args(); random.seed(42); np.random.seed(42); torch.manual_seed(42)
    dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rec, skipped=match_features(a.feature_root, read_labels(a.train_list))
    print(f"Matched {len(rec)} training features; unmatched feature files: {skipped}")
    if len(rec)<20: raise RuntimeError("Too few features matched. Check feature/list paths and filename stems.")
    ds=FeatureDataset(rec,a.max_segments); nv=max(1,int(.15*len(ds)))
    tr,va=random_split(ds,[len(ds)-nv,nv],generator=torch.Generator().manual_seed(42))
    tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=0)
    vl=DataLoader(va,batch_size=a.batch_size,num_workers=0)
    m=TemporalTransformerSummarizer(max_segments=a.max_segments).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4); loss_fn=nn.BCEWithLogitsLoss()
    best=-1; out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    for ep in range(1,a.epochs+1):
        m.train(); losses=[]
        for x,y,n in tl:
            x,y,n=x.to(dev),y.to(dev),n.to(dev); opt.zero_grad()
            logits,_=m(x,n); loss=loss_fn(logits,y); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); losses.append(loss.item())
        m.eval(); ys=[]; ps=[]; vls=[]
        with torch.no_grad():
            for x,y,n in vl:
                x,y,n=x.to(dev),y.to(dev),n.to(dev); z,_=m(x,n)
                vls.append(loss_fn(z,y).item()); ys.extend(y.cpu().numpy().astype(int)); ps.extend(torch.sigmoid(z).cpu().numpy())
        pred=(np.array(ps)>=.5).astype(int); auc=roc_auc_score(ys,ps) if len(set(ys))>1 else 0
        print(f"epoch {ep:02d}: train_loss={np.mean(losses):.4f} val_loss={np.mean(vls):.4f} val_auc={auc:.4f} acc={accuracy_score(ys,pred):.4f} f1={f1_score(ys,pred,zero_division=0):.4f}")
        if auc>best:
            best=auc; torch.save({"model_state":m.state_dict(),"config":{"max_segments":a.max_segments},"epoch":ep,"val_auc":auc},out)
            print("  saved",out)
    print("Training complete. Keep the independent test set untouched until final evaluation.")
if __name__=="__main__": main()
