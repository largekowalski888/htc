from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from htc import Config, LabelMapping


def truth_index_for_row(row: pd.Series, inf_truth: dict, mapping: LabelMapping) -> int:
    truth_type = inf_truth.get('type', 'constant')
    if truth_type == 'constant':
        return mapping.name_to_index(inf_truth['value'])
    if truth_type == 'column':
        return mapping.name_to_index(str(row[inf_truth['column']]))
    raise ValueError(f'Unsupported truth type: {truth_type}')


def main(args):
    run_dir = Path(args.run_dir)
    task = json.loads((run_dir / 'task.json').read_text(encoding='utf-8'))
    config = Config(run_dir / 'config.json')
    mapping = LabelMapping.from_config(config)
    n_classes = len(mapping)

    pred_path = run_dir / 'predictions.csv'
    if not pred_path.exists():
        raise SystemExit('predictions.csv not found. Run predict_task.py first.')
    df = pd.read_csv(pred_path)
    if df.empty:
        raise SystemExit('predictions.csv is empty.')

    inf_truth = task['inference_source']['truth']
    rows_by_subject = {}
    qc_rows = []

    for _, row in df.iterrows():
        subject_name = row['subject_name']
        y_true = np.array([truth_index_for_row(row, inf_truth, mapping)], dtype=np.int64)
        y_pred = np.array([int(row['pred_label_index'])], dtype=np.int64)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        acc = accuracy_score(y_true, y_pred)

        rows_by_subject.setdefault(subject_name, []).append({
            'timestamp': row['timestamp'],
            'condition': row.get('condition', ''),
            'confusion_matrix': cm,
            'accuracy': float(acc),
        })
        qc_rows.append({
            'subject_name': subject_name,
            'timestamp': row['timestamp'],
            'condition': row.get('condition', ''),
            'true_label_index': int(y_true[0]),
            'true_label_name': mapping.label_names()[int(y_true[0])],
            'pred_label_index': int(y_pred[0]),
            'pred_label_name': mapping.label_names()[int(y_pred[0])],
            'pred_confidence': row.get('pred_confidence', None),
            'accuracy': float(acc),
            'status': 'ok',
        })

    subject_rows = []
    for subject_name, items in rows_by_subject.items():
        cms = np.stack([it['confusion_matrix'] for it in items], axis=0)
        subject_cm = cms.sum(axis=0)
        mean_acc = float(np.mean([it['accuracy'] for it in items]))
        subject_rows.append({
            'subject_name': subject_name,
            'image_names': [it['timestamp'] for it in items],
            'n_images': len(items),
            'condition': items[0]['condition'],
            'confusion_matrix': subject_cm,
            'accuracy': mean_acc,
        })

    df_test = pd.DataFrame(subject_rows)
    df_test.to_pickle(run_dir / 'test_table.pkl.xz')
    pd.DataFrame(qc_rows).to_csv(run_dir / 'qc_report.csv', index=False)

    print(f"Saved: {run_dir / 'test_table.pkl.xz'}")
    print(f"Saved: {run_dir / 'qc_report.csv'}")
    print(f"Subjects evaluated: {len(df_test)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build HTC-style test_table.pkl.xz from predictions')
    parser.add_argument('--run-dir', required=True, type=str, help='Directory produced by train_task.py / predict_task.py')
    main(parser.parse_args())
