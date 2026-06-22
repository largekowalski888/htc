from pathlib import Path
import json
import numpy as np
import pandas as pd

output_dir = Path(r"C:\DKFZ\outputs\dummy_confusion_run")
output_dir.mkdir(parents=True, exist_ok=True)

# Use a very small dummy 3-class example
labels = {
        "class_a": 0,
        "class_b": 1,
        "class_c": 2
    }

# Per-subject confusion matrices
cm1 = np.array([
    [40,  2,  1],
    [ 3, 35,  2],
    [ 1,  4, 30],
], dtype=np.int64)

cm2 = np.array([
    [38,  4,  0],
    [ 2, 33,  5],
    [ 0,  3, 31],
], dtype=np.int64)

rows = [
    {"subject_name": "subject_001", "confusion_matrix": cm1, "accuracy": 0.88},
    {"subject_name": "subject_002", "confusion_matrix": cm2, "accuracy": 0.86},
]

df = pd.DataFrame(rows)
df.to_pickle(output_dir / "test_table.pkl.xz")

# Minimal config to satisfy the notebook path
config = {
    "label_mapping": labels
}

with open(output_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(f"Saved: {output_dir / 'test_table.pkl.xz'}")
print(f"Saved: {output_dir / 'config.json'}")
