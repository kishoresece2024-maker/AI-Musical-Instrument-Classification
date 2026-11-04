# AI-Based Musical Instrument Classification Using Audio Signals

**Student:** [Your name]  
**Department:** [Department / college]  
**Guide:** [Guide name]  
**Academic year:** [Year]

## Abstract
This project develops a machine-learning pipeline to identify a musical instrument from an audio clip. Audio is converted to mono at a fixed sample rate, trimmed for silence, and represented by MFCCs, spectral centroid, zero-crossing rate, chroma, spectral rolloff, and RMS summaries. Support Vector Machine (SVM) and Random Forest classifiers are compared using validation macro-F1. The selected classifier is refitted on the training and validation sets and evaluated once on a held-out test set using accuracy, macro precision, macro recall, macro F1, a classification report, and a confusion matrix.

## 1. Introduction
Instrument recognition is an audio classification problem used in music information retrieval, audio indexing, and interactive music applications. Timbre, harmonic content, attack, and spectral shape provide cues that help distinguish instruments.

## 2. Problem statement
Given a labeled audio dataset, learn a mapping from audio-derived features to instrument labels and predict the label of an unseen audio clip.

## 3. Objectives
1. Organize licensed audio into labeled instrument classes.
2. Preprocess audio consistently.
3. Extract MFCC and spectral features.
4. Train and compare SVM and Random Forest.
5. Evaluate unseen clips with standard classification metrics.
6. Demonstrate prediction on an uploaded audio file.

## 4. System requirements
Python 3.10–3.12; NumPy; Pandas; Librosa; scikit-learn; Matplotlib; Joblib; Streamlit; SoundFile.

## 5. Methodology
### 5.1 Dataset
Dataset name: [Fill in]  
Source and citation: [Fill in]  
License/usage terms: [Fill in]  
Instrument classes: [Fill in]  
Number of usable clips per class: [Copy from prepare_dataset output]

### 5.2 Preprocessing
Each clip is loaded as mono audio at 22,050 Hz and limited to five seconds. Silence is trimmed; very short clips are padded before feature computation. Clips that cannot be decoded or produce invalid features are skipped with a warning.

### 5.3 Feature extraction
- **MFCCs:** represent the short-term spectral envelope on the mel scale.
- **Spectral centroid:** estimates the spectral “center of mass” and relates to brightness.
- **Zero-crossing rate:** counts sign changes and can reflect noisiness/percussive content.
- **Chroma:** summarizes energy across 12 pitch classes.
- **Spectral rolloff:** estimates the frequency below which a chosen proportion of spectral energy lies.
- **RMS energy:** summarizes signal energy/loudness.

Mean and standard deviation over frames provide fixed-length features per clip.

### 5.4 Models
- **SVM with RBF kernel:** uses standardized features and a nonlinear decision boundary.
- **Random Forest:** combines many decision trees and can model nonlinear feature relationships.
Validation macro-F1 selects the model. The held-out test set is not used for selection.

### 5.5 Evaluation
Report accuracy, macro precision, macro recall, macro F1, per-class precision/recall/F1, and the confusion matrix. Macro metrics weight each class equally.

## 6. Results
**Do not invent results.** After running the pipeline, copy actual values from `reports/test_metrics.json` and `reports/classification_report.csv`.

| Metric | Measured result |
|---|---:|
| Accuracy | [Fill from test_metrics.json] |
| Macro precision | [Fill from test_metrics.json] |
| Macro recall | [Fill from test_metrics.json] |
| Macro F1-score | [Fill from test_metrics.json] |
| Selected model | [Fill from test_metrics.json] |

Insert `reports/confusion_matrix.png` and discuss which instruments are confused and plausible reasons.

## 7. Discussion
Performance can be affected by dataset size, class balance, room acoustics, recording equipment, instrument technique, background noise, and whether clips contain solo instruments or mixtures. A file-level split reduces direct duplication across sets, but multiple clips from the same source recording/session can still create leakage; group-based splitting is preferable when source/session metadata exists.

## 8. Limitations
The current baseline uses summary features and traditional classifiers. It may struggle with polyphonic music, overlapping instruments, noisy recordings, and instruments not represented during training. A classifier always chooses among known classes; it does not reliably detect an unknown instrument.

## 9. Future enhancements
- Group-based evaluation by song/session/performer.
- Add data augmentation (carefully, without crossing splits).
- Compare a CNN using log-mel spectrograms.
- Add unknown-class rejection and more instrument categories.
- Deploy the Streamlit demo.

## 10. Conclusion
The project implements an end-to-end audio classification workflow from licensed labeled data to feature extraction, model selection, held-out testing, and interactive prediction. Final conclusions must be based on the measured results obtained from the chosen dataset.

## References
1. McFee, B. et al. “librosa: Audio and Music Signal Analysis in Python.” Proceedings of the Python in Science Conference, 2015.
2. Pedregosa, F. et al. “Scikit-learn: Machine Learning in Python.” Journal of Machine Learning Research, 2011.
3. [Add the selected dataset's official citation and license page.]
