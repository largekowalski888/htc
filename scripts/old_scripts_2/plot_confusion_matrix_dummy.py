
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc.models.common.MetricAggregationClassification import MetricAggregationClassification

# Use forward slashes to avoid Windows escape warnings
run_dir = Path("C:/DKFZ/outputs/dummy_confusion_run")

config = Config(run_dir / "config.json")
mapping = LabelMapping.from_config(config)

print("Loaded label names:", mapping.label_names())

agg = MetricAggregationClassification(
    run_dir / "test_table.pkl.xz",
    config,
    metrics=["accuracy", "confusion_matrix"]
)

df_metrics = agg.subject_metrics()

cm = np.stack(df_metrics["confusion_matrix"].values)
cm_abs = np.sum(cm, axis=0)

print(df_metrics)
print(cm_abs)

# ---- Normalize rows to percentages ----
cm_rel = cm_abs / cm_abs.sum(axis=1, keepdims=True)
cm_rel = np.nan_to_num(cm_rel) * 100

labels = mapping.label_names()

# ---- Plot heatmap ----
fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)

sns.heatmap(
    cm_rel,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"label": "%"},
    ax=ax,
)

ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("Dummy Confusion Matrix")

# Save figure
plt.savefig(run_dir / "dummy_confusion_matrix.png", dpi=200)

# Show figure
plt.show()


"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

run_dir = Path(r"C:\DKFZ\htc\outputs\dummy_confusion_run")

df = pd.read_pickle(run_dir / "test_table.pkl.xz")

with open(run_dir / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

labels = config["label_mapping"]

cm = np.stack(df["confusion_matrix"].values)
cm_abs = np.sum(cm, axis=0)

# Row-normalized confusion matrix in %
cm_rel = cm_abs / cm_abs.sum(axis=1, keepdims=True)
cm_rel = np.nan_to_num(cm_rel) * 100

fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
sns.heatmap(
    cm_rel,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"label": "%"},
    ax=ax,
)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("Dummy Confusion Matrix")
plt.show()

print(f"Mean accuracy: {df['accuracy'].mean():.3f} ± {df['accuracy'].std():.3f}")
"""
