# Transformer-Based Video Anomaly Detection and Summarization

## Overview
A Transformer-based temporal model classifies UCF-Crime videos as normal
or anomalous using pre-extracted C3D features and generates shortened
video summaries using segment-level scores.

## Dataset
- Dataset: UCF-Crime
- Test videos: 290
- Normal videos: 150
- Anomalous videos: 140
- Input feature dimension: 4096

## Architecture
- Layer normalization and linear feature projection
- Learnable positional embeddings
- 3-layer Transformer encoder
- 256-dimensional model, 8 attention heads
- Segment prediction head
- Video-level prediction using top-25%-logit pooling

## Test Results
- Accuracy: 77.93%
- ROC-AUC: 0.8797
- Anomaly precision: 89.58%
- Anomaly recall: 61.43%
- Anomaly F1-score: 0.7288
- Classification threshold: 0.50

Confusion matrix:
                 Predicted Normal  Predicted Anomaly
Actual Normal          140                10
Actual Anomaly          54                86

## Outputs
- Trained checkpoint: checkpoints/transformer_best.pt
- Test predictions: transformer_test_results.csv
- Segment diagnostics: segment_diagnostics.csv
- Summary videos and reports: summaries/
- Mixed-batch score analysis: summaries/mixed_batch/score_analysis.csv

## Limitations
The model is trained with video-level labels. Segment scores do not prove
that the selected segments contain the actual anomalous event. Threshold
selection requires separate validation data, and precise temporal
localization has not been independently established.

## Conclusion
The implemented system demonstrates video-level anomaly discrimination
and generates shortened video summaries. Temporal localization and
validation-based threshold selection remain future improvements.
