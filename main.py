from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import wfdb
import urllib.request
from pathlib import Path
from scipy.signal import resample_poly
import traceback

from src.config import CHECKPOINTS_DIR, DATA_RAW_DIR, CLASSES, IDX_TO_CLASS, WIN_BEFORE, WIN_AFTER, AAMI_MAP, CLASS_TO_IDX
from src.dataset import bandpass_filter
from src.model import ChronoQRSNet

app = FastAPI(
    title="ChronoQRS API",
    description="Inter-Patient ECG Arrhythmia Classification Backend",
    version="1.0.0"
)

model = None
device = torch.device("cpu")

@app.on_event("startup")
def load_model():
    global model
    ckpt_path = CHECKPOINTS_DIR / "best_chrono_qrs.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run training first.")
    
    model = ChronoQRSNet(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[INFO] ChronoQRSNet loaded successfully into FastAPI backend!")


class PredictionRequest(BaseModel):
    record_id: str = "200"
    database: str = "mitdb"


def ensure_local_record(record_id: str, database: str):
    """Ensures record files exist locally (downloads from S3 mirror if missing)."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # S3 mirror base depending on database
    s3_base = "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0" if database == "mitdb" else "https://physionet-open.s3.amazonaws.com/incartdb/1.0.0"
    
    for ext in ["hea", "dat", "atr"]:
        dest = DATA_RAW_DIR / f"{record_id}.{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{s3_base}/{record_id}.{ext}"
        try:
            print(f"[INFO] Downloading {record_id}.{ext} from AWS S3 mirror...")
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"[WARNING] S3 download failed for {ext}: {e}")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "model": "ChronoQRSNet",
        "classes": CLASSES,
        "docs": "/docs"
    }


@app.post("/predict")
def predict_record(req: PredictionRequest):
    try:
        # Ensure files are downloaded locally via fast S3 mirror
        ensure_local_record(req.record_id, req.database)
        
        local_path = str(DATA_RAW_DIR / req.record_id)
        rec = wfdb.rdrecord(local_path)
        ann = wfdb.rdann(local_path, "atr")
        
        raw_sig = rec.p_signal[:, 0]
        fs = float(rec.fs)

        if req.database == "incartdb" and fs != 360.0:
            raw_sig = resample_poly(raw_sig, up=360, down=257)
            fs = 360.0

        filtered_sig = bandpass_filter(raw_sig)

        valid_peaks, valid_labels = [], []
        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol in AAMI_MAP:
                new_sample = int(round(sample * (360.0 / fs))) if req.database == "incartdb" else sample
                valid_peaks.append(new_sample)
                valid_labels.append(AAMI_MAP[symbol])

        valid_peaks = np.array(valid_peaks)
        if len(valid_peaks) < 3:
            raise HTTPException(status_code=400, detail="Not enough valid heartbeats found in record.")

        rr_intervals = np.diff(valid_peaks) / fs
        waveforms, rr_features, true_labels = [], [], []

        for i in range(1, len(valid_peaks) - 1):
            peak = valid_peaks[i]
            if peak < WIN_BEFORE or peak + WIN_AFTER >= len(filtered_sig):
                continue

            beat_wave = filtered_sig[peak - WIN_BEFORE : peak + WIN_AFTER]
            beat_wave = (beat_wave - np.mean(beat_wave)) / (np.std(beat_wave) + 1e-8)

            pre_rr = rr_intervals[i - 1]
            post_rr = rr_intervals[i]
            local_avg_rr = np.mean(rr_intervals[max(0, i - 10) : i + 1])

            rr_feat = [
                pre_rr, post_rr,
                pre_rr / (local_avg_rr + 1e-8),
                post_rr / (local_avg_rr + 1e-8)
            ]

            waveforms.append(beat_wave)
            rr_features.append(rr_feat)
            true_labels.append(CLASS_TO_IDX[valid_labels[i]])

        X_w = torch.tensor(np.array(waveforms, dtype=np.float32)).unsqueeze(1).to(device)
        X_r = torch.tensor(np.array(rr_features, dtype=np.float32)).to(device)

        with torch.no_grad():
            logits = model(X_w, X_r)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        class_counts = {c: int(np.sum(preds == idx)) for idx, c in enumerate(CLASSES)}
        total_beats = int(len(preds))

        return {
            "record_id": req.record_id,
            "database": req.database,
            "total_beats_evaluated": total_beats,
            "prediction_counts": class_counts
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[ERROR TRACEBACK]:\n{error_trace}")
        raise HTTPException(status_code=500, detail=str(e))