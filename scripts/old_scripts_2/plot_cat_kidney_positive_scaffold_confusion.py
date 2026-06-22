from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc.models.common.MetricAggregationClassification import MetricAggregationClassification

run_dir = Path(r"C:\DKFZ\outputs\cat_kidney_positive_scaffold")

config = Config(run_dir / 'config.json')
mapping = LabelMapping.from_config(config)

agg = MetricAggregationClassification(
    run_dir / 'test_table.pkl.xz',
    config,
    metrics=['accuracy', 'confusion_matrix']
)

df_metrics = agg.subject_metrics()
cm = np.stack(df_metrics['confusion_matrix'].values)
cm_abs = np.sum(cm, axis=0)
cm_rel = cm_abs / np.clip(cm_abs.sum(axis=1, keepdims=True), 1, None)
cm_rel = np.nan_to_num(cm_rel) * 100
labels = mapping.label_names()

fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
sns.heatmap(cm_rel, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': '%'}, ax=ax)
ax.set_xlabel('predicted')
ax.set_ylabel('true')
ax.set_title('CAT Kidney-Positive Scaffold Confusion Matrix')
plt.savefig(run_dir / 'cat_kidney_positive_scaffold_confusion_matrix.png', dpi=200)
plt.show()

print(df_metrics[['subject_name', 'accuracy']].head())
print('Mean accuracy:', df_metrics['accuracy'].mean())
