from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

for d in [DATA_RAW_DIR, DATA_PROC_DIR, FIGURES_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42

# Signal Processing Constants (MIT-BIH is 360 Hz)
FS = 360
LOWCUT = 0.5
HIGHCUT = 40.0
FILTER_ORDER = 3

# Beat Windowing: 0.25s (90 samples) before R-peak, 0.45s (162 samples) after
WIN_BEFORE = int(0.25 * FS)  # 90 samples
WIN_AFTER = int(0.45 * FS)   # 162 samples
BEAT_LEN = WIN_BEFORE + WIN_AFTER  # 252 samples

# Standard AAMI EC57 Class Mapping
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",  # Normal
    "A": "S", "a": "S", "J": "S", "S": "S",            # Supraventricular ectopic
    "V": "V", "E": "V",                                # Ventricular ectopic
    "F": "F",                                          # Fusion
    "/": "Q", "f": "Q", "Q": "Q"                       # Unknown / Paced
}

CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# de Chazal et al. Inter-Patient Split (excludes paced records 102, 104, 107, 217)
DS1_TRAIN = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"
]

DS2_TEST = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"
]