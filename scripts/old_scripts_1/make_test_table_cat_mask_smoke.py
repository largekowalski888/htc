
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------------------
# CONFIG
# -----------------------------------------
run_dir = Path(r"C:\DKFZ\outputs\cat_mask_smoke_run")
run_dir.mkdir(parents=True, exist_ok=True)

mask_path = Path(r"C:\DKFZ\MLData\data\Cat_Pig\Cat_atlas\Cat_0007_kidney\data\P044_Pig003_G001_2020_02_01\2020_02_01_10_23_29\_hypergui_1\mask.csv")

subject_name = "CAT_sample_001"   # change if you want
image_name = "CAT_image_001"      # change if you want

# -----------------------------------------
# LOAD MASK
# -----------------------------------------
mask = pd.read_csv(mask_path, header=None).to_numpy(dtype=np.int64)

# Ground truth: flatten binary mask
y_true = mask.reshape(-1)

# Perfect smoke-test predictions
y_pred = y_true.copy()

# Optional: add 5% controlled noise later
# rng = np.random.default_rng(42)
# n_flip = max(1, int(len(y_pred) * 0.05))
# flip_idx = rng.choice(len(y_pred), size=n_flip, replace=False)
# y_pred[flip_idx] = 1 - y_pred[flip_idx]

# -----------------------------------------
# METRICS
# -----------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
acc = accuracy_score(y_true, y_pred)

rows = [{
    "subject_name": subject_name,
    "image_name": image_name,
    "confusion_matrix": cm,
    "accuracy": float(acc),
}]

df = pd.DataFrame(rows)
df.to_pickle(run_dir / "test_table.pkl.xz")

# Minimal config for binary label mapping
config = {
    "label_mapping": {
        "background": 0,
        "roi": 1
    }
}

with open(run_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("Saved:", run_dir / "test_table.pkl.xz")
print("Saved:", run_dir / "config.json")
print("Confusion matrix:\n", cm)
print("Accuracy:", acc)
