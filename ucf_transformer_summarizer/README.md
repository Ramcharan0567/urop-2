# UCF-Crime Transformer Extractive Summarizer

Separate from the existing LSTM/MIL project. It uses a Transformer encoder over temporal C3D features (expected shape: segments x 4096), learns video-level anomaly scoring with top-k pooling, ranks segments, and uses FFmpeg to assemble a short summary MP4.

## Setup
```bash
cd ~/Desktop
unzip ucf_transformer_summarizer.zip
cd ucf_transformer_summarizer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
brew install ffmpeg  # only if ffmpeg is missing
```

Find existing label files:
```bash
find ~/Desktop/archive -name 'Anomaly_Train.txt' -o -name 'Anomaly_Test.txt'
```
Use those paths below; do not redownload the dataset. Existing feature root from prior work:
`$HOME/Desktop/archive/Extracted C3D Features/Extracted C3D Features`

## Train
```bash
python train.py --feature-root "$HOME/Desktop/archive/Extracted C3D Features/Extracted C3D Features" --train-list "/ABSOLUTE/PATH/Anomaly_Train.txt" --epochs 20 --batch-size 8
```

## Evaluate once on the held-out test list
```bash
python evaluate.py --feature-root "$HOME/Desktop/archive/Extracted C3D Features/Extracted C3D Features" --test-list "/ABSOLUTE/PATH/Anomaly_Test.txt"
```

## Create an extractive summary
Find the original MP4 and its matching feature file, then:
```bash
python summarize.py --video "/path/to/original.mp4" --features "/path/to/matching_feature.npy" --checkpoint checkpoints/transformer_best.pt --ratio 0.20 --out outputs/summary.mp4
```
Writes the summary MP4 and `outputs/summary.segments.json`.

## Scientific caveat
The labels are video-level normal/anomaly labels, not human-authored summaries or segment timestamps. The segment scores are weakly supervised, not ground-truth event boundaries. The summary script assumes features are uniformly distributed over the video's duration; confirm the original C3D extraction sampling before treating times as exact. The current script selects top-scoring segments; it does not yet learn semantic importance or narrative coherence.
