import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from src.config import (
    DATA_PROC_DIR, FS, LOWCUT, HIGHCUT, FILTER_ORDER,
    WIN_BEFORE, WIN_AFTER, AAMI_MAP, CLASS_TO_IDX, DS1_TRAIN, DS2_TEST
)


def bandpass_filter(signal: np.ndarray) -> np.ndarray:
    """Apply zero-phase 0.5-40 Hz Butterworth bandpass filter."""
    b, a = butter(FILTER_ORDER, [LOWCUT, HIGHCUT], btype="band", fs=FS)
    return filtfilt(b, a, signal)


def extract_record_beats(record_id: str):
    """
    Stream a single MIT-BIH record from PhysioNet, filter lead 0 (MLII),
    and extract 252-sample beat waveforms + 4 RR-interval features.
    """
    rec = wfdb.rdrecord(record_id, pn_dir="mitdb")
    ann = wfdb.rdann(record_id, "atr", pn_dir="mitdb")

    sig = bandpass_filter(rec.p_signal[:, 0])

    # Keep only valid AAMI heartbeat annotations before computing RR intervals
    valid_peaks = []
    valid_Labels = []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in AAMI_MAP:
            valid_peaks.append(sample)
            valid_Labels.append(AAMI_MAP[symbol])

    valid_peaks = np.array(valid_peaks)
    rr_intervals = np.diff(valid_peaks) / float(FS)  # RR in seconds

    waveforms, rr_features, labels = [], [], []

    # Skip first and last beat so every beat has a valid pre-RR and post-RR
    for i in range(1, len(valid_peaks) - 1):
        peak = valid_peaks[i]
        if peak < WIN_BEFORE or peak + WIN_AFTER >= len(sig):
            continue

        # 1. Extract 252-sample waveform and z-score normalize per beat
        beat_wave = sig[peak - WIN_BEFORE : peak + WIN_AFTER]
        beat_wave = (beat_wave - np.mean(beat_wave)) / (np.std(beat_wave) + 1e-8)

        # 2. Extract temporal RR features (in seconds & normalized by local rhythm)
        pre_rr = rr_intervals[i - 1]
        post_rr = rr_intervals[i]
        
        # Local 10-beat running average RR to make features patient-rate invariant
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
        labels.append(CLASS_TO_IDX[valid_Labels[i]])

    return (
        np.array(waveforms, dtype=np.float32),
        np.array(rr_features, dtype=np.float32),
        np.array(labels, dtype=np.int64),
    )


def build_split_dataset(record_list, split_name: str, force_rebuild: bool = False):
    """Download, process, and cache DS1 (train) or DS2 (test) splits to disk."""
    cache_path = DATA_PROC_DIR / f"{split_name}.npz"
    if cache_path.exists() and not force_rebuild:
        print(f"[INFO] Loading cached {split_name} from {cache_path}")
        data = np.load(cache_path)
        return data["X_wave"], data["X_rr"], data["y"], data["patient_ids"]

    all_wave, all_rr, all_y, all_pids = [], [], [], []
    print(f"[INFO] Building {split_name} from PhysioNet MIT-BIH ({len(record_list)} records)...")

    for rec_id in tqdm(record_list, desc=split_name):
        X_w, X_r, y = extract_record_beats(rec_id)
        all_wave.append(X_w)
        all_rr.append(X_r)
        all_y.append(y)
        all_pids.extend([rec_id] * len(y))

    X_wave = np.concatenate(all_wave, axis=0)
    X_rr = np.concatenate(all_rr, axis=0)
    y = np.concatenate(all_y, axis=0)
    patient_ids = np.array(all_pids)

    np.savez_compressed(
        cache_path, X_wave=X_wave, X_rr=X_rr, y=y, patient_ids=patient_ids
    )
    print(f"[SUCCESS] Saved {split_name}: {len(y)} beats -> {cache_path}")
    return X_wave, X_rr, y, patient_ids


class MITBIHDataset(Dataset):
    """PyTorch Dataset returning (waveform, rr_features, label)."""
    def __init__(self, X_wave: np.ndarray, X_rr: np.ndarray, y: np.ndarray):
        self.X_wave = torch.from_numpy(X_wave).unsqueeze(1)  # Shape: (N, 1, 252)
        self.X_rr = torch.from_numpy(X_rr)                   # Shape: (N, 4)
        self.y = torch.from_numpy(y)                         # Shape: (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_wave[idx], self.X_rr[idx], self.y[idx]


if __name__ == "__main__":
    build_split_dataset(DS1_TRAIN, "DS1_train")
    build_split_dataset(DS2_TEST, "DS2_test")