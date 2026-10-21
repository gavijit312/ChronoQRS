# ⚡ ChronoQRS: Inter-Patient ECG Arrhythmia Classification API

[![API Status](https://img.shields.io/badge/API-Live-brightgreen)](https://chronoqrs.onrender.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-teal)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://www.docker.com/)

**ChronoQRS** is an enterprise-grade deep learning microservice designed for automated inter-patient electrocardiogram (ECG) arrhythmia classification. Built using a dual-branch neural architecture and deployed via a cloud-hosted FastAPI backend, it processes morphological waveforms and RR-interval dynamics to classify heartbeats in accordance with AAMI clinical standards.

---

## 🚀 Live Production Endpoints

* **API Status & Health Check:** [https://chronoqrs.onrender.com](https://chronoqrs.onrender.com)
* **Interactive Swagger Documentation:** [https://chronoqrs.onrender.com/docs](https://chronoqrs.onrender.com/docs)

---

## 🧠 System Architecture & Methodology

1. **Signal Processing Pipeline:** Utilizes `wfdb` and SciPy for physiological signal parsing, zero-phase Butterworth bandpass filtering, and dynamic resampling.
2. **Feature Engineering:** Combines raw windowed heartbeat morphology with precise local RR-interval feature ratios (pre-RR, post-RR, and local averages).
3. **Dual-Branch Deep Learning Model (`ChronoQRSNet`):** Built with PyTorch to process spatial waveform features and temporal rhythm dynamics simultaneously.
4. **Production Microservice:** Built with **FastAPI**, containerized using **Docker**, and deployed on cloud infrastructure to serve real-time cardiac inference.

---

## 🛠️ Tech Stack

* **Deep Learning:** PyTorch, NumPy, SciPy, WFDB
* **Backend & API:** FastAPI, Uvicorn, Pydantic
* **Deployment & DevOps:** Docker, Render Cloud Infrastructure, Git/GitHub

---

## 💻 Local Quickstart

If you want to run the backend locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/gavijit312/ChronoQRS.git](https://github.com/gavijit312/ChronoQRS.git)
   cd ChronoQRS