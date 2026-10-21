import argparse, json, shutil, subprocess, tempfile, re
from pathlib import Path
import numpy as np, torch
from data import load_feature
from model import TemporalTransformerSummarizer

def run(cmd):
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",required=True); p.add_argument("--features",required=True)
    p.add_argument("--checkpoint",default="checkpoints/transformer_best.pt")
    p.add_argument("--ratio",type=float,default=.20); p.add_argument("--out",default="outputs/summary.mp4")
    a=p.parse_args()
    video=Path(a.video).expanduser().resolve(); fp=Path(a.features).expanduser().resolve(); out=Path(a.out).expanduser().resolve()
    if not video.exists() or not fp.exists(): raise FileNotFoundError("Video or feature file not found.")
    ff=shutil.which("ffmpeg")
    if not ff: raise RuntimeError("Install FFmpeg first: brew install ffmpeg")
    if not 0<a.ratio<1: raise ValueError("--ratio must be between 0 and 1")
    dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    c=torch.load(a.checkpoint,map_location="cpu",weights_only=False); ms=c["config"]["max_segments"]
    m=TemporalTransformerSummarizer(max_segments=ms); m.load_state_dict(c["model_state"]); m.to(dev).eval()
    x=load_feature(fp)
    if x.shape[1]!=4096: raise ValueError(f"Expected 4096 feature dimensions, got {x.shape}")
    if len(x)>ms: x=x[np.linspace(0,len(x)-1,ms).round().astype(int)]
    n=len(x)
    xp=np.concatenate([x,np.zeros((ms-n,x.shape[1]),np.float32)]) if n<ms else x
    with torch.no_grad():
        _,s=m(torch.from_numpy(xp)[None].to(dev),torch.tensor([n],device=dev))
        scores=torch.sigmoid(s[0,:n]).cpu().numpy()
    probe=subprocess.run([ff,"-i",str(video),"-hide_banner"],capture_output=True,text=True)
    mt=re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)",probe.stderr)
    if not mt: raise RuntimeError("Could not read video duration.")
    duration=int(mt[1])*3600+int(mt[2])*60+float(mt[3])
    # Rank short temporal windows using a moving average. This favors
    # coherent neighboring segments while keeping the same summary budget.
    k = max(1, min(n, round(n * a.ratio)))
    window = min(3, n)
    smooth = np.convolve(scores, np.ones(window) / window, mode="same")
    chosen = []
    available = np.ones(n, dtype=bool)
    while len(chosen) < k:
        candidates = np.where(available)[0]
        if len(candidates) == 0:
            break
        center = int(candidates[np.argmax(smooth[candidates])])
        lo = max(0, center - window // 2)
        hi = min(n, lo + min(window, k - len(chosen)))
        lo = max(0, hi - min(window, k - len(chosen)))
        block = [i for i in range(lo, hi) if available[i]]
        if not block:
            available[center] = False
            continue
        chosen.extend(block)
        # Avoid selecting overlapping windows.
        for j in range(max(0, lo - window + 1), min(n, hi + window - 1)):
            available[j] = False
    if len(chosen) < k:
        remaining = [i for i in np.argsort(scores)[::-1] if i not in chosen]
        chosen.extend(remaining[:k-len(chosen)])
    chosen = sorted(chosen[:k])
    out.parent.mkdir(parents=True,exist_ok=True); intervals=[]
    with tempfile.TemporaryDirectory(prefix="ucf_summary_") as td:
        td=Path(td); clips=[]
        for j,i in enumerate(chosen):
            start=duration*i/n; end=duration*(i+1)/n; clip=td/f"clip_{j:03d}.mp4"
            run([ff,"-y","-ss",f"{start:.4f}","-i",str(video),"-t",f"{max(.05,end-start):.4f}",
                 "-an","-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2","-c:v","libx264","-preset","veryfast","-crf","23","-pix_fmt","yuv420p",str(clip)])
            clips.append(clip); intervals.append({"segment":i+1,"start_seconds":round(start,3),"end_seconds":round(end,3),"anomaly_score":round(float(scores[i]),6)})
        listing=td/"concat.txt"; listing.write_text("".join(f"file '{q}'\n" for q in clips))
        run([ff,"-y","-f","concat","-safe","0","-i",str(listing),"-c","copy",str(out)])
    report=out.with_suffix(".segments.json")
    all_segments = [
        {
            "segment": i + 1,
            "start_seconds": round(duration * i / n, 3),
            "end_seconds": round(duration * (i + 1) / n, 3),
            "anomaly_score": round(float(scores[i]), 6),
            "selected": i in chosen,
        }
        for i in range(n)
    ]
    report.write_text(json.dumps({
        "video": str(video),
        "duration_seconds": duration,
        "assumption": "Features are mapped uniformly across the video duration.",
        "warning": "Weakly supervised scores are not verified event boundaries.",
        "segments": intervals,
        "all_segment_scores": all_segments,
    }, indent=2))
    print("Summary:", out); print("Segment report:", report)
    for z in intervals: print(f"segment {z['segment']:02d}: {z['start_seconds']:.2f}-{z['end_seconds']:.2f}s score={z['anomaly_score']:.4f}")
if __name__=="__main__": main()
