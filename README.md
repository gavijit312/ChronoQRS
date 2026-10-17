# ChronoQRS: Inter-Patient ECG Arrhythmia Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8%25%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ChronoQRS** is a deep learning pipeline for automated heartbeat arrhythmia classification on the standard **MIT-BIH Arrhythmia Database**. Unlike typical academic or tutorial notebooks that leak patient morphology across random splits to claim 99% accuracy, ChronoQRS enforces a strict **zero-leakage inter-patient split** (*de Chazal et al.*), ensuring evaluation on completely unseen clinical cohorts.

---

## 🔬 Key Architectural Differentiators

1. **Zero-Leakage Inter-Patient Split:** Separates the 44 non-paced records into two disjoint cohorts—22 patients for training (`DS1`) and 22 entirely different patients for testing (`DS2`). No patient data ever crosses between training and evaluation.
2. **Dual-Branch Spatio-Temporal Fusion:**
   - **Morphology Branch (`wave_branch`):** A multi-scale 1D Convolutional Neural Network processing 252-sample beat windows (0.25s pre-peak, 0.45s post-peak) filtered with a zero-phase $0.5\text{--}40\text{ Hz}$ Butterworth bandpass.
   - **Chrono Branch (`rr_branch`):** A dedicated dense projection modeling local and raw RR-interval dynamics (`pre_RR`, `post_RR`, and patient-rate normalized ratios) to prevent timing signals from being washed out by waveform convolutions.
3. **Class-Weighted Optimization:** Employs smoothed square-root inverse frequency loss weighting to handle severe class imbalance without destabilizing precision on normal beats.

---

## 📂 Repository Structure

```text
chrono-qrs/
├── data/                    # Raw PhysioNet records & processed .npz caches (ignored by git)
├── notebooks/               # Signal exploration & filtering validation
├── results/
│   ├── checkpoints/         # Saved PyTorch model weights (best_chrono_qrs.pt)
│   ├── figures/             # Confusion matrices and color-coded ECG strips
│   └── metrics_table.md     # Auto-generated performance reports
├── src/
│   ├── config.py            # AAMI mappings, filter constants, patient splits
│   ├── dataset.py           # S3 parallel downloader, filtering, windowing, & RR extraction
│   ├── baseline.py          # Random Forest baseline on RR-interval dynamics
│   ├── model.py             # ChronoQRSNet dual-branch architecture
│   ├── train.py             # Class-weighted training loop & validation checkpointing
│   ├── evaluate.py          # Unseen-patient DS2 evaluation & confusion matrix rendering
│   └── visualize.py         # Clinical ECG strip renderer color-coded by prediction
├── requirements.txt
└── README.md