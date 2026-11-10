"""PAMAP2 protocol loader and subject-independent splitting."""
from pathlib import Path
import numpy as np
import pandas as pd

ACTIVITIES = {1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running", 6: "cycling"}
FEATURES = ["heart_rate", "hand_acc", "chest_acc", "ankle_acc"]

def load_pamap2(root, activities=None):
    """Read subject*.dat (54 columns), derive accelerometer magnitudes.
    Missing features are imputed using TRAINING statistics in prepare_split.
    """
    root = Path(root)
    paths = sorted(root.rglob("subject*.dat"))
    if not paths:
        raise FileNotFoundError(f"No PAMAP2 subject*.dat files under {root}")
    parts = []
    for path in paths:
        raw = pd.read_csv(path, sep=r"\s+", header=None, na_values=["NaN"], engine="c")
        if raw.shape[1] != 54:
            raise ValueError(f"{path}: expected 54 columns, got {raw.shape[1]}")
        subject = int(path.stem.replace("subject", ""))
        out = pd.DataFrame({"subject": subject, "timestamp": raw.iloc[:, 0], "activity": raw.iloc[:, 1].astype(int), "heart_rate": raw.iloc[:, 2]})
        # PAMAP2 IMU 16g accelerometer columns: hand 4:7, chest 21:24, ankle 38:41 (zero-indexed).
        for name, start in (("hand_acc", 4), ("chest_acc", 21), ("ankle_acc", 38)):
            out[name] = np.sqrt(raw.iloc[:, start:start+3].pow(2).sum(axis=1, min_count=3))
        out = out[out["activity"].isin(activities or ACTIVITIES.keys())]
        parts.append(out)
    return pd.concat(parts, ignore_index=True)

def prepare_split(data, test_subjects=(108, 109), validation_subjects=(107,)):
    """Split by subject to avoid temporal/subject leakage. Fit imputation on train only."""
    train = data[~data.subject.isin((*test_subjects, *validation_subjects))].copy()
    val = data[data.subject.isin(validation_subjects)].copy()
    test = data[data.subject.isin(test_subjects)].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Empty split: supply data for train, validation and test subjects")
    medians = train[FEATURES].median().fillna(0)
    for frame in (train, val, test):
        frame[FEATURES] = frame[FEATURES].fillna(medians)
    return train, val, test
