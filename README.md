# 🫀 ChronoQRS: Inter-Patient ECG Arrhythmia Classification System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-005571?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 🌟 Executive Summary

Automated ECG analysis often suffers from **over-optimistic generalization** due to random intra-patient splitting (where heartbeats from the same patient exist in both training and test sets). **ChronoQRS** solves this by implementing strict **inter-patient partitioning** across diverse clinical databases (MIT-BIH and St. Petersburg INCART).

The system couples a custom dual-branch neural network (`ChronoQRSNet`) with an asynchronous **FastAPI** backend optimized via AWS S3-mirrored local caching, wrapped entirely inside an isolated **Docker** container for reproducible production deployment.

---

## 🏗️ System Architecture

```text
                                  [ PhysioNet AWS S3 Mirror ]
                                               │
                                               ▼
[ Client / UI ] ──(POST /predict)──> [ FastAPI Backend (Uvicorn) ]
                                               │
                                               ├─> wfdb (Signal & Annotation Parsing)
                                               ├─> SciPy (Bandpass Filtering & Resampling)
                                               │
                                               ▼
                              [ ChronoQRSNet (Dual-Branch PyTorch) ]
                              ├── Branch 1: 1D CNN (Morphological Waveform)
                              └── Branch 2: Dense MLP (Dynamic RR-Intervals)
                                               │
                                               ▼
                                  [ AAMI EC57 Class Output ]
```

---

## 📂 Project Structure

```text
ChronoQRS/
├── src/
│   ├── __init__.py
│   ├── config.py         # Global hyperparameters, paths, and ANSI/AAMI mappings
│   ├── dataset.py        # Signal preprocessing, bandpass filtering, and windowing
│   ├── model.py          # ChronoQRSNet PyTorch dual-branch architecture
│   └── evaluate.py       # Clinical evaluation metrics & confusion matrix generation
├── results/
│   └── checkpoints/      # Trained model weights (best_chrono_qrs.pt)
├── data/                 # Local raw cache directory (auto-populated via S3 mirror)
├── Dockerfile            # Container definition for Python/PyTorch/FastAPI
├── .dockerignore         # Excluded build artifacts & venv files
├── main.py               # FastAPI application entrypoint & S3 fast-download handler
└── requirements.txt      # Pinned production dependencies
```

---

## 📊 Clinical Taxonomy & Performance

Model predictions conform to the **ANSI/AAMI EC57** standard, mapping raw arrhythmia annotations into five clinically distinct heartbeat categories:

| Class Code | Description | Clinical Significance |
| :---: | :--- | :--- |
| **N** | Normal / Bundle Branch Block | Baseline sinus rhythm & normal conduction variants |
| **S** | Supraventricular Ectopic | Premature atrial contractions (PACs) / nodal beats |
| **V** | Ventricular Ectopic | Premature ventricular contractions (PVCs) — high risk |
| **F** | Fusion Beats | Combination of normal and ventricular beats |
| **Q** | Unknown / Unclassifiable | Unrecognized morphology / paced beats |

---

## ⚙️ Local Installation & Development

### 1. Clone the Repository
```bash
git clone https://github.com/gavijit312/ChronoQRS.git
cd ChronoQRS
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Development Server
```bash
uvicorn main:app --reload
```

Navigate to **`http://127.0.0.1:8000/docs`** to explore the interactive Swagger UI documentation.

---

## 🐳 Docker Production Deployment

Containerize and execute the backend in a fully isolated environment using Docker:

### 1. Build the Docker Image
```bash
docker build -t chrono-qrs-api .
```

### 2. Run the Container (Detached Mode)
```bash
docker run -d -p 8000:8000 --name chrono-container chrono-qrs-api
```

### 3. Stop or Clean Up Container
```bash
docker stop chrono-container
docker rm chrono-container
```

---

## 🔌 API Reference & Usage

### Endpoint: `POST /predict`

Evaluates an entire ECG record by streaming/caching records from the PhysioNet S3 mirror, filtering signals, extracting windows, and running inference through `ChronoQRSNet`.

**Request Body:**
```json
{
  "record_id": "200",
  "database": "mitdb"
}
```

**Response Payload:**
```json
{
  "record_id": "200",
  "database": "mitdb",
  "total_beats_evaluated": 2601,
  "prediction_counts": {
    "N": 1942,
    "S": 0,
    "V": 659,
    "F": 0,
    "Q": 0
  }
}
```

---

## 🛡️ License

Distributed under the **MIT License**. See `LICENSE` for more information.