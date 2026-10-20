import urllib.request
import numpy as np
import wfdb
import torch
from torch.utils.data import DataLoader
from scipy.signal import resample_poly
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.config import (
    CLASSES, CHECKPOINTS_DIR, DATA_RAW_DIR, WIN_BEFORE, WIN_AFTER, AAMI_MAP, CLASS_TO_IDX
)
from src.dataset import bandpass_filter, MITBIHDataset
from src.model import ChronoQRSNet

# Direct AWS S3 Open Data mirror for St. Petersburg INCART Database
INCART_S3_BASE = "https://physionet-open.s3.amazonaws.com/incartdb/1.0.0"

INCART_TEST_RECORDS = ["I01"]


def download_incart_record(record_id: str):
    """Fast-download INCART files (.hea, .dat, .atr) from AWS S3 mirror."""
    for ext in ["hea", "dat", "atr"]:
        dest = DATA_RAW_DIR / f"{record_id}.{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{INCART_S3_BASE}/{record_id}.{ext}"
        try:
            print(f"[INFO] Downloading {record_id}.{ext} from AWS S3...")
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"[WARNING] S3 download failed for {ext}: {e}")


def process_incart_record(record_id: str):
    """
    Fast-download, resample from 257 Hz to 360 Hz, filter, and extract 
    252-sample windows from an INCART record.
    """
    download_incart_record(record_id)
    print(f"[INFO] Processing INCART record {record_id} from local disk...")
    
    local_path = str(DATA_RAW_DIR / record_id)
    try:
        rec = wfdb.rdrecord(local_path)
        ann = wfdb.rdann(local_path, "atr")
    except Exception as e:
        print(f"[ERROR] Could not read local record {record_id}: {e}")
        return None, None, None

    # INCART native fs is 257 Hz, target fs is 360 Hz
    orig_fs = 257.0
    target_fs = 360.0
    
    raw_sig = rec.p_signal[:, 0]

    # 1. Resample signal from 257 Hz to 360 Hz
    resampled_sig = resample_poly(raw_sig, up=360, down=257)
    filtered_sig = bandpass_filter(resampled_sig)

    # 2. Rescale annotation sample indices to 360 Hz time base
    scale_factor = target_fs / orig_fs
    valid_peaks, valid_labels = [], []
    
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in AAMI_MAP:
            new_sample = int(round(sample * scale_factor))
            valid_peaks.append(new_sample)
            valid_labels.append(AAMI_MAP[symbol])

    valid_peaks = np.array(valid_peaks)
    if len(valid_peaks) < 3:
        return None, None, None

    rr_intervals = np.diff(valid_peaks) / target_fs

    waveforms, rr_features, labels = [], [], []

    for i in range(1, len(valid_peaks) - 1):
        peak = valid_peaks[i]
        if peak < WIN_BEFORE or peak + WIN_AFTER >= len(filtered_sig):
            continue

        beat_wave = filtered_sig[peak - WIN_BEFORE : peak + WIN_AFTER]
        beat_wave = (beat_wave - np.mean(beat_wave)) / (np.std(beat_wave) + 1e-8)

        pre_rr = rr_intervals[i - 1]
        post_rr = rr_intervals[i]
        start_idx = max(0, i - 10)
        local_avg_rr = np.mean(rr_intervals[start_idx : i + 1])

        rr_feat = [
            pre_rr,
            post_rr,
            pre_rr / (local_avg_rr + 1e-8),
            post_rr / (local_avg_rr + 1e-8),
        ]

        waveforms.append(beat_wave)
        rr_features.append(rr_feat)
        labels.append(CLASS_TO_IDX[valid_labels[i]])

    return (
        np.array(waveforms, dtype=np.float32),
        np.array(rr_features, dtype=np.float32),
        np.array(labels, dtype=np.int64),
    )


def evaluate_incart():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CHECKPOINTS_DIR / "best_chrono_qrs.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}. Train the model first via `python -m src.train`.")

    model = ChronoQRSNet(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_waves, all_rrs, all_ys = [], [], []

    for rec_id in INCART_TEST_RECORDS:
        X_w, X_r, y = process_incart_record(rec_id)
        if X_w is not None and len(X_w) > 0:
            all_waves.append(X_w)
            all_rrs.append(X_r)
            all_ys.append(y)

    if not all_waves:
        print("[ERROR] No INCART records were successfully processed.")
        return

    X_wave_test = np.concatenate(all_waves, axis=0)
    X_rr_test = np.concatenate(all_rrs, axis=0)
    y_test = np.concatenate(all_ys, axis=0)

    test_ds = MITBIHDataset(X_wave_test, X_rr_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for x_w, x_r, _ in test_loader:
            x_w, x_r = x_w.to(device), x_r.to(device)
            logits = model(x_w, x_r)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    y_pred = np.array(all_preds)

    macro_f1_nsv = f1_score(y_test, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    
    print("\n" + "=" * 65)
    print("CROSS-DATASET EVALUATION: CHRONOQRS ON INCART RECORD I01")
    print("=" * 65)
    print(classification_report(
        y_test, y_pred, labels=range(len(CLASSES)), target_names=CLASSES, digits=4, zero_division=0
    ))
    print(f"External Cross-Dataset Clinical Macro F1 (N, S, V): {macro_f1_nsv:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=range(len(CLASSES))))
    print("=" * 65)


if __name__ == "__main__":
    evaluate_incart()
    