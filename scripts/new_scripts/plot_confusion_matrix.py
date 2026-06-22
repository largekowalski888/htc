from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from htc.evaluation.metrics.scores import normalize_grouped_cm
from htc.models.common.MetricAggregationClassification import MetricAggregationClassification
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc.utils.helper_functions import sort_labels, sort_labels_cm


def main(args):
    run_dir = Path(args.run_dir)
    config = Config(run_dir / 'config.json')
    mapping = LabelMapping.from_config(config)

    agg = MetricAggregationClassification(run_dir / 'test_table.pkl.xz', config, metrics=['accuracy', 'confusion_matrix'])
    df_metrics = agg.subject_metrics()
    cm = np.stack(df_metrics['confusion_matrix'].values)
    cm_abs = np.sum(cm, axis=0)
    cm_rel, cm_rel_std = normalize_grouped_cm(cm)
    cm_rel *= 100
    cm_rel_std *= 100

    original_labels = mapping.label_names()
    try:
        labels = sort_labels(original_labels)
        cm_abs = sort_labels_cm(cm_abs, cm_order=original_labels, target_order=labels).astype(np.int64)
        cm_rel = sort_labels_cm(cm_rel, cm_order=original_labels, target_order=labels)
        cm_rel_std = sort_labels_cm(cm_rel_std, cm_order=original_labels, target_order=labels)
    except Exception:
        labels = original_labels

    text = np.vectorize(lambda x_abs, x_rel, x_std: '' if x_abs == 0 else f'{x_rel:.1f}%\n({x_std:.1f}%)')(cm_abs, cm_rel, cm_rel_std)

    fig, ax = plt.subplots(figsize=(16, 12), tight_layout=True)
    sns.heatmap(cm_rel, annot=text, fmt='s', annot_kws={'size': 9}, cmap='Blues', xticklabels=labels, yticklabels=labels, cbar_kws={'label': '%'}, ax=ax)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(args.title or f'Confusion Matrix: {run_dir.name}')
    plt.xticks(rotation=70)
    plt.yticks(rotation=0)
    out_path = run_dir / 'confusion_matrix.png'
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(df_metrics[['subject_name', 'accuracy']].head())
    print('Mean accuracy:', df_metrics['accuracy'].mean())
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot confusion matrix for a run directory')
    parser.add_argument('--run-dir', required=True, type=str)
    parser.add_argument('--title', default='', type=str)
    main(parser.parse_args())
