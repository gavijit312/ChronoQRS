import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.config import CLASSES, DS1_TRAIN, DS2_TEST, SEED, RESULTS_DIR
from src.dataset import build_split_dataset


def run_rr_baseline():
    # 1. Load inter-patient splits
    _, X_rr_train, y_train, _ = build_split_dataset(DS1_TRAIN, "DS1_train")
    _, X_rr_test, y_test, _ = build_split_dataset(DS2_TEST, "DS2_test")

    print("\n[INFO] Training Random Forest Baseline on RR-interval dynamics...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_rr_train, y_train)

    # 2. Evaluate on unseen patients (DS2)
    y_pred = rf.predict(X_rr_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 60)
    print("BASELINE RESULTS: RANDOM FOREST (RR-INTERVALS ONLY) ON DS2")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred, labels=range(len(CLASSES)), target_names=CLASSES, digits=4, zero_division=0
    ))
    print(f"Overall Unseen-Patient Macro F1: {macro_f1:.4f}")
    print("\nConfusion Matrix (Rows: True, Cols: Pred):")
    print(confusion_matrix(y_test, y_pred, labels=range(len(CLASSES))))

    # Save baseline metrics for README comparison
    out_file = RESULTS_DIR / "baseline_metrics.txt"
    with open(out_file, "w") as f:
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(classification_report(
            y_test, y_pred, labels=range(len(CLASSES)), target_names=CLASSES, digits=4, zero_division=0
        ))
    print(f"\n[INFO] Saved baseline report to {out_file}")


if __name__ == "__main__":
    run_rr_baseline()