import numpy as np
import wfdb
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.config import (
    DATA_RAW_DIR, FIGURES_DIR, CHECKPOINTS_DIR, FS,
    WIN_BEFORE, WIN_AFTER, AAMI_MAP, CLASS_TO_IDX, IDX_TO_CLASS, CLASSES
)
from src.dataset import bandpass_filter, ensure_records_downloaded
from src.model import ChronoQRSNet

# Color palette for AAMI classes on clinical ECG strip
CLASS_COLORS = {
    "N": "#1f77b4",  # Clean Blue for Normal
    "S": "#ff7f0e",  # Vivid Orange for Supraventricular
    "V": "#d62728",  # Crimson Red for Ventricular
    "F": "#9467bd",  # Purple for Fusion
    "Q": "#7f7f7f",  # Grey for Unknown
}


def plot_annotated_ecg_strip(record_id: str = "200", start_sec: float = 12.0, duration_sec: float = 10.0):
    """
    Plot an ECG strip from an unseen DS2 patient with each heartbeat colored by
    its ChronoQRSNet prediction and annotated with Pred vs. True labels.
    """
    ensure_records_downloaded([record_id])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model checkpoint
    ckpt_path = CHECKPOINTS_DIR / "best_chrono_qrs.pt"
    model = ChronoQRSNet(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Load and filter continuous ECG signal
    local_record_path = str(DATA_RAW_DIR / record_id)
    rec = wfdb.rdrecord(local_record_path)
    ann = wfdb.rdann(local_record_path, "atr")
    sig = bandpass_filter(rec.p_signal[:, 0])

    # Extract all valid peaks across the record for accurate RR calculation
    valid_peaks, valid_syms = [], []
    for sample, sym in zip(ann.sample, ann.symbol):
        if sym in AAMI_MAP:
            valid_peaks.append(sample)
            valid_syms.append(AAMI_MAP[sym])
    valid_peaks = np.array(valid_peaks)
    rr_intervals = np.diff(valid_peaks) / float(FS)

    start_sample = int(start_sec * FS)
    end_sample = int((start_sec + duration_sec) * FS)

    # 3. Set up clinical ECG plot
    fig, ax = plt.subplots(figsize=(15, 4.8), dpi=150)
    time_axis = np.arange(start_sample, end_sample) / float(FS)

    # Subtle ECG grid background
    ax.plot(
        time_axis, sig[start_sample:end_sample],
        color="#cfd8dc", linewidth=1.0, label="Baseline Filtered ECG", zorder=1
    )

    # 4. Predict and color each beat inside the time window
    for i in range(1, len(valid_peaks) - 1):
        peak = valid_peaks[i]
        if peak - WIN_BEFORE < start_sample or peak + WIN_AFTER > end_sample:
            continue

        true_cls = valid_syms[i]
        beat_raw = sig[peak - WIN_BEFORE : peak + WIN_AFTER]
        beat_norm = (beat_raw - np.mean(beat_raw)) / (np.std(beat_raw) + 1e-8)

        pre_rr = rr_intervals[i - 1]
        post_rr = rr_intervals[i]
        local_avg_rr = np.mean(rr_intervals[max(0, i - 10) : i + 1])
        rr_feat = np.array(
            [pre_rr, post_rr, pre_rr / (local_avg_rr + 1e-8), post_rr / (local_avg_rr + 1e-8)],
            dtype=np.float32
        )

        with torch.no_grad():
            x_w = torch.from_numpy(beat_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            x_r = torch.from_numpy(rr_feat).unsqueeze(0).to(device)
            logits = model(x_w, x_r)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_cls = IDX_TO_CLASS[pred_idx]

        # Overlay colored beat waveform
        beat_time = np.arange(peak - WIN_BEFORE, peak + WIN_AFTER) / float(FS)
        color = CLASS_COLORS[pred_cls]
        ax.plot(beat_time, beat_raw, color=color, linewidth=2.2, zorder=3)

        # Annotate R-peak
        r_time = peak / float(FS)
        r_amp = sig[peak]
        is_correct = (pred_cls == true_cls)
        tag = f"{pred_cls}" if is_correct else f"Pred:{pred_cls}\n(True:{true_cls})"
        box_color = color if is_correct else "#000000"

        ax.scatter([r_time], [r_amp], color=color, s=35, zorder=4)
        ax.annotate(
            tag,
            xy=(r_time, r_amp),
            xytext=(r_time, r_amp + 0.32),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=box_color, alpha=0.88, lw=0)
        )

    ax.set_title(
        f"ChronoQRS Beat-by-Beat Arrhythmia Triage — Unseen Patient Record #{record_id} ({start_sec:.0f}s–{start_sec + duration_sec:.0f}s)",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("MLII Lead Amplitude (mV)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#eceff1")
    ax.set_ylim(np.min(sig[start_sample:end_sample]) - 0.4, np.max(sig[start_sample:end_sample]) + 0.75)

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color=CLASS_COLORS["N"], lw=2.5, label="Predicted Normal (N)"),
        Line2D([0], [0], color=CLASS_COLORS["S"], lw=2.5, label="Predicted Supraventricular (S)"),
        Line2D([0], [0], color=CLASS_COLORS["V"], lw=2.5, label="Predicted Ventricular (V)"),
        Line2D([0], [0], color=CLASS_COLORS["F"], lw=2.5, label="Predicted Fusion (F)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    out_path = FIGURES_DIR / f"ecg_strip_patient_{record_id}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved color-coded ECG strip -> {out_path}")


if __name__ == "__main__":
    # Record 200 (unseen DS2 patient) has alternating Normal (N) and Ventricular (V) bigeminy/trigeminy beats
    plot_annotated_ecg_strip(record_id="200", start_sec=12.0, duration_sec=10.0)
    # Record 222 (unseen DS2 patient) has Supraventricular (S) atrial arrhythmias
    plot_annotated_ecg_strip(record_id="222", start_sec=360.0, duration_sec=10.0)