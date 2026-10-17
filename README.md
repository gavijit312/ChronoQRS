# 🫀 ChronoQRS: ECG Arrhythmia Classification & Triage Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ChronoQRS** is an enterprise-grade, high-performance deep learning system designed for automated electrocardiogram (ECG) arrhythmia classification and clinical triage. Built with a dual-branch neural architecture, the system simultaneously processes raw morphological waveforms and inter-beat (RR) intervals to achieve near-perfect diagnostic accuracy while maintaining low-latency inference through a containerized FastAPI backend.

---

## 🚀 Key Features

- **Dual-Branch Deep Learning Architecture**: Combines convolutional feature extraction for heartbeat morphology with temporal modeling for RR-interval dynamics.
- **Inter-Patient Evaluation Split**: Validated using rigorous inter-patient testing partitions to ensure clinical generalizability and prevent patient data leakage.
- **High-Performance Production API**: Powered by FastAPI with automated Swagger documentation and low-overhead request validation.
- **Cloud-Ready Caching**: Integrated with fast S3 mirror caching for high-throughput diagnostic retrieval.
- **Fully Containerized**: Packaged into a reproducible, lightweight Docker container for frictionless cross-platform deployment.

---

## 🛠️ Tech Stack

- **Deep Learning Framework**: PyTorch, torchvision, torchaudio
- **Numerical & Scientific Computing**: NumPy, SciPy, Scikit-Learn
- **API Framework**: FastAPI, Uvicorn
- **Containerization**: Docker
- **Version Control & Management**: Git, Python venv

---

## 📦 Model Weights & Checkpoints

Due to GitHub's file size limits for standard tracking, the pre-trained model weights (`best_chrono_qrs.pt`) require proper placement before running local inference or Docker builds.

To set up your weights:
1. Ensure your trained model checkpoint file is placed in the project directory structure at:
   ```text
   results/checkpoints/best_chrono_qrs.pt
   ```

---

## ⚙️ Getting Started & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/gavijit312/ChronoQRS.git
cd ChronoQRS
```

### 2. Set Up a Virtual Environment (Optional for Local Dev)
```bash
python -m venv venv
# On Windows:
venv\Scripts\Activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🐳 Docker Deployment

To run ChronoQRS in an isolated production container:

### 1. Build the Docker Image
```bash
docker build -t chrono-qrs-api .
```

### 2. Run the Container
```bash
docker run -d -p 8000:8000 --name chrono-container chrono-qrs-api
```

Once running, access the interactive Swagger API documentation in your browser at:
👉 **`http://127.0.0.1:8000/docs`**

---

## 🔌 API Endpoints Overview

- `GET /`: Health check and system status.
- `POST /predict`: Accepts ECG signal arrays/features and returns real-time arrhythmia classification probabilities and triage decisions.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.