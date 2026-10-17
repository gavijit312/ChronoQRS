# ChronoQRS Evaluation Summary (DS2 Unseen Patients)

## Per-Class Performance on DS2

| Class | Description | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **N** | Normal | 0.9685 | 0.9343 | 0.9511 | 44,218 |
| **S** | Supraventricular (SVEB) | 0.3409 | 0.1629 | 0.2204 | 1,836 |
| **V** | Ventricular (VEB) | 0.7420 | 0.9348 | 0.8273 | 3,219 |
| **F** | Fusion | 0.0000 | 0.0000 | 0.0000 | 388 |
| **Q** | Unknown | 0.0000 | 0.0000 | 0.0000 | 7 |

- **Macro F1 (Primary N, S, V):** `0.6663`
- **Macro F1 (All 5 Classes):** `0.3998`

## Ready-to-Use Resume Line

> Built an ECG arrhythmia classifier (ChronoQRS) with a dual-branch 1D CNN on the MIT-BIH database using the de Chazal inter-patient split, reaching 0.67 clinical macro F1 (0.40 5-class macro F1) and 93.5% recall on ventricular beats across 22 unseen test patients.
