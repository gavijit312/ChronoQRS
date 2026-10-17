import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.config import (
    CLASSES, DS1_TRAIN, CHECKPOINTS_DIR, SEED
)
from src.dataset import build_split_dataset, MITBIHDataset
from src.model import ChronoQRSNet, count_parameters

# Hyperparameters optimized for fast CPU / single-GPU convergence
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Hold out 3 DS1 patients for leak-free validation checkpoint selection
VAL_PATIENTS = {"114", "119", "208"}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Compute smoothed inverse-frequency weights: (total / count)^0.5
    Prevents extreme gradient explosions on ultra-rare classes (F, Q) while
    heavily penalizing missed Supraventricular (S) and Ventricular (V) beats.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    # Replace 0 counts with 1 to avoid division by zero
    counts = np.maximum(counts, 1.0)
    weights = np.sqrt(np.max(counts) / counts)
    # Cap maximum weight to prevent destabilizing normal beat precision
    weights = np.clip(weights, 1.0, 12.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Load cached DS1 dataset
    X_wave, X_rr, y, pids = build_split_dataset(DS1_TRAIN, "DS1_train")

    # 2. Split DS1 by patient into Train (19 patients) and Val (3 patients)
    val_mask = np.isin(pids, list(VAL_PATIENTS))
    train_mask = ~val_mask

    train_ds = MITBIHDataset(X_wave[train_mask], X_rr[train_mask], y[train_mask])
    val_ds = MITBIHDataset(X_wave[val_mask], X_rr[val_mask], y[val_mask])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    print(f"[INFO] DS1 Split -> Train: {len(train_ds):,} beats (19 patients) | Val: {len(val_ds):,} beats (3 patients)")

    # 3. Initialize model, class-weighted loss, and optimizer
    model = ChronoQRSNet(num_classes=len(CLASSES)).to(device)
    print(f"[INFO] ChronoQRSNet initialized with {count_parameters(model):,} trainable parameters")

    class_weights = compute_class_weights(y[train_mask], len(CLASSES), device)
    print(f"[INFO] Class weights {CLASSES}: {np.round(class_weights.cpu().numpy(), 2)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_f1 = 0.0
    best_ckpt_path = CHECKPOINTS_DIR / "best_chrono_qrs.pt"

    print("\n" + "=" * 75)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val Macro F1 (N,S,V)':<22} | {'Status'}")
    print("=" * 75)

    for epoch in range(1, EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for x_w, x_r, labels in train_loader:
            x_w, x_r, labels = x_w.to(device), x_r.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(x_w, x_r)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_w, x_r, labels in val_loader:
                x_w, x_r, labels = x_w.to(device), x_r.to(device), labels.to(device)
                logits = model(x_w, x_r)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_ds)
        # Track Macro F1 on the primary clinical classes present in Val (N=0, S=1, V=2)
        val_f1_nsv = f1_score(all_targets, all_preds, labels=[0, 1, 2], average="macro", zero_division=0)

        status = ""
        if val_f1_nsv > best_val_f1:
            best_val_f1 = val_f1_nsv
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1_nsv": best_val_f1,
                "class_weights": class_weights.cpu().numpy(),
            }, best_ckpt_path)
            status = "★ Saved Best"

        print(f"{epoch:02d}/{EPOCHS:02d}    | {train_loss:<12.4f} | {val_loss:<12.4f} | {val_f1_nsv:<22.4f} | {status}")

    print("=" * 75)
    print(f"[SUCCESS] Training complete! Best checkpoint saved to: {best_ckpt_path} (Val F1: {best_val_f1:.4f})")


if __name__ == "__main__":
    train_model()