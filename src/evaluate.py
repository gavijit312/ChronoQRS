import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, recall_score, precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    CLASSES, DS2_TEST, CHECKPOINTS_DIR, FIGURES_DIR, RESULTS_DIR
)
from src.dataset import build_split_dataset, MITBIHDataset
from src.model import ChronoQRSNet


def plot_confusion_matrix(cm: np.ndarray, save_path):
    """Render both raw count and row-normalized (%) confusion matrices side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # 1. Raw Beat Counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0], cbar=False
    )
    axes[0].set_title("DS2 Unseen Patients — Raw Beat Counts", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted AAMI Class")
    axes[0].set_ylabel("True Cardiologist Class")

    # 2. Row-Normalized Recall (%)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
    sns.heatmap(
        cm_norm, annot=True, fmt=".1f", cmap="YlGnBu",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1], cbar=True
    )
    axes[1].set_title("DS2 Unseen Patients — Row-Normalized Recall (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted AAMI Class")
    axes[1].set_ylabel("True Cardiologist Class")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix plot -> {save_path}")


def evaluate_on_ds2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CHECKPOINTS_DIR / "best_chrono_qrs.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run `python -m src.train` first.")

    # 1. Load DS2 unseen patient test set
    X_wave_test, X_rr_test, y_test, _ = build_split_dataset(DS2_TEST, "DS2_test")
    test_ds = MITBIHDataset(X_wave_test, X_rr_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # 2. Load trained ChronoQRSNet checkpoint
    model = ChronoQRSNet(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Inference on DS2
    all_preds = []
    with torch.no_grad():
        for x_w, x_r, _ in test_loader:
            x_w, x_r = x_w.to(device), x_r.to(device)
            logits = model(x_w, x_r)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    y_pred = np.array(all_preds)

    # 4. Compute Metrics
    macro_f1_all = f1_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1_nsv = f1_score(y_test, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    recalls = recall_score(y_test, y_pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    precisions = precision_score(y_test, y_pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    f1s = f1_score(y_test, y_pred, labels=range(len(CLASSES)), average=None, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASSES)))
    report = classification_report(
        y_test, y_pred, labels=range(len(CLASSES)), target_names=CLASSES, digits=4, zero_division=0
    )

    print("\n" + "=" * 65)
    print("HONEST INTER-PATIENT EVALUATION: CHRONOQRS 1D-CNN ON DS2 (22 PATIENTS)")
    print("=" * 65)
    print(report)
    print(f"Macro F1 (5-Class AAMI N,S,V,F,Q): {macro_f1_all:.4f}")
    print(f"Macro F1 (Primary Clinical N,S,V): {macro_f1_nsv:.4f}")
    print(f"Ventricular (V) Recall:            {recalls[2] * 100:.2f}% (Precision: {precisions[2] * 100:.2f}%)")
    print(f"Supraventricular (S) Recall:       {recalls[1] * 100:.2f}% (Precision: {precisions[1] * 100:.2f}%)")
    print("\nConfusion Matrix (Rows: True, Cols: Pred):")
    print(cm)

    # 5. Save Plots and Markdown Summary
    cm_fig_path = FIGURES_DIR / "ds2_confusion_matrix.png"
    plot_confusion_matrix(cm, cm_fig_path)

    metrics_md_path = RESULTS_DIR / "metrics_table.md"
    resume_line = (
        f'Built an ECG arrhythmia classifier (ChronoQRS) with a dual-branch 1D CNN on the MIT-BIH '
        f'database using the de Chazal inter-patient split, reaching {macro_f1_nsv:.2f} clinical macro F1 '
        f'({macro_f1_all:.2f} 5-class macro F1) and {recalls[2] * 100:.1f}% recall on ventricular beats '
        f'across 22 unseen test patients.'
    )

    with open(metrics_md_path, "w") as f:
        f.write("# ChronoQRS Evaluation Summary (DS2 Unseen Patients)\n\n")
        f.write("## Per-Class Performance on DS2\n\n")
        f.write("| Class | Description | Precision | Recall | F1-Score | Support |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        desc = ["Normal", "Supraventricular (SVEB)", "Ventricular (VEB)", "Fusion", "Unknown"]
        supports = cm.sum(axis=1)
        for i, c in enumerate(CLASSES):
            f.write(f"| **{c}** | {desc[i]} | {precisions[i]:.4f} | {recalls[i]:.4f} | {f1s[i]:.4f} | {supports[i]:,} |\n")
        f.write(f"\n- **Macro F1 (Primary N, S, V):** `{macro_f1_nsv:.4f}`\n")
        f.write(f"- **Macro F1 (All 5 Classes):** `{macro_f1_all:.4f}`\n\n")
        f.write("## Ready-to-Use Resume Line\n\n")
        f.write(f"> {resume_line}\n")

    print("\n" + "=" * 65)
    print("YOUR RESUME LINE (Also saved to results/metrics_table.md):")
    print("=" * 65)
    print(resume_line)
    print("=" * 65)


if __name__ == "__main__":
    evaluate_on_ds2()