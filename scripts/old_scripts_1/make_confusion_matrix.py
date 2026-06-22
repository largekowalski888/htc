import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from htc.evaluation.metrics.scores import normalize_grouped_cm
from htc.models.common.MetricAggregationClassification import MetricAggregationClassification
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc.utils.helper_functions import sort_labels, sort_labels_cm


def plot_confusion_matrix(run_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(run_dir / "config.json")
    mapping = LabelMapping.from_config(config)

    agg = MetricAggregationClassification(
        run_dir / "test_table.pkl.xz",
        config,
        metrics=["accuracy", "confusion_matrix"]
    )

    df_metrics = agg.subject_metrics()

    cm = np.stack(df_metrics["confusion_matrix"].values)
    cm_abs = np.sum(cm, axis=0)

    cm_rel, cm_rel_std = normalize_grouped_cm(cm)
    cm_rel *= 100
    cm_rel_std *= 100

    original_labels = mapping.label_names()
    labels = sort_labels(original_labels)

    cm_abs = sort_labels_cm(cm_abs, cm_order=original_labels, target_order=labels)
    cm_rel = sort_labels_cm(cm_rel, cm_order=original_labels, target_order=labels)
    cm_rel_std = sort_labels_cm(cm_rel_std, cm_order=original_labels, target_order=labels)

    text = np.vectorize(
        lambda x_abs, x_rel, x_std: "" if x_abs == 0 else f"{x_rel:.1f}%\n({x_std:.1f}%)"
    )(cm_abs, cm_rel, cm_rel_std)

    fig, ax = plt.subplots(figsize=(26, 15), dpi=100)

    sns.heatmap(
        cm_rel.T,
        annot=text.T,
        fmt="s",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        cbar_kws={"label": "%"},
        ax=ax
    )

    ax.set_xlabel("True label", fontweight="bold")
    ax.set_ylabel("Predicted label", fontweight="bold")
    ax.set_title("Confusion Matrix")

    plt.xticks(rotation=70)
    plt.tight_layout()

    fig.savefig(output_dir / "confusion_matrix.png")
    fig.savefig(output_dir / "confusion_matrix.pdf")

    print(f"Accuracy: {df_metrics['accuracy'].mean():.3f} ± {df_metrics['accuracy'].std():.3f}")
    print(f"Saved confusion matrix to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/confusion_matrix"))

    args = parser.parse_args()

    plot_confusion_matrix(args.run_dir, args.output_dir)
    
"""
Run it like this in cmd:
    python scripts/make_confusion_matrix.py \
  --run-dir outputs/baseline_run \
  --output-dir outputs/baseline_run/confusion_matrix
  
Run folder must contain:
    config.json
    test_table.pkl.xz
    
====================================================================

OLD CODE:
    python scripts/make_confusion_matrix.py \
  --run-dir path/to/training_dir/median_pixel/YOUR_RUN_NAME \
  --output-dir outputs/cm_test
"""