from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from data_adapters import load_heipor_table, build_heipor_label_space, load_custom_table


def build_model(model_cfg: dict):
    model_type = model_cfg.get('type', 'random_forest')
    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=int(model_cfg.get('n_estimators', 300)),
            max_depth=model_cfg.get('max_depth', None),
            random_state=int(model_cfg.get('random_state', 42)),
            n_jobs=-1,
            class_weight=model_cfg.get('class_weight', 'balanced_subsample'),
        )
    if model_type == 'logistic_regression':
        return LogisticRegression(
            max_iter=int(model_cfg.get('max_iter', 1000)),
            C=float(model_cfg.get('C', 1.0)),
            class_weight=model_cfg.get('class_weight', 'balanced'),
            multi_class='auto',
        )
    if model_type == 'linear_svm':
        return LinearSVC(
            C=float(model_cfg.get('C', 1.0)),
            class_weight=model_cfg.get('class_weight', 'balanced'),
            random_state=int(model_cfg.get('random_state', 42)),
        )
    raise ValueError(f'Unsupported model type: {model_type}')


def main(args):
    task_path = Path(args.task)
    task = json.loads(task_path.read_text(encoding='utf-8'))

    out_dir = Path(args.output_dir) if args.output_dir else Path(task['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    label_space = build_heipor_label_space(task)
    train_cfg = task['training_source']
    adapter = train_cfg['adapter']

    if adapter == 'heipor_table':
        X_train, y_train, meta_df, data_meta = load_heipor_table(
            train_cfg['path'],
            feature_column=train_cfg['feature']['column'],
            label_column=train_cfg['label']['column'],
            label_space=label_space,
            label_transform=train_cfg['label'].get('transform', 'identity'),
            normalize_features=bool(train_cfg['feature'].get('normalize', False)),
        )
    elif adapter == 'custom_table':
        X_train, y_train, meta_df, data_meta = load_custom_table(
            train_cfg['path'],
            feature_column=train_cfg['feature']['column'],
            label_column=train_cfg['label']['column'],
            group_column=train_cfg.get('group_column'),
            normalize_features=bool(train_cfg['feature'].get('normalize', False)),
        )
        # overwrite label space for explicit custom labels if needed
        if task.get('label_space', {}).get('source') == 'explicit_labels':
            pass
    else:
        raise ValueError(f'Unsupported training adapter: {adapter}')

    model = build_model(task.get('model', {}))
    print(f"Training task '{task['task_name']}' on {X_train.shape[0]} rows x {X_train.shape[1]} features")
    model.fit(X_train, y_train)

    with open(out_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(out_dir / 'task.json', 'w', encoding='utf-8') as f:
        json.dump(task, f, indent=2)
    with open(out_dir / 'label_map.json', 'w', encoding='utf-8') as f:
        json.dump({'labels': label_space['labels'], 'name_to_index': label_space['name_to_index']}, f, indent=2)
    label_space['config'].save_config(out_dir / 'config.json')

    run_info = {
        'task_name': task['task_name'],
        'output_dir': str(out_dir),
        'training_rows': int(X_train.shape[0]),
        'feature_length': int(X_train.shape[1]),
        'label_space': label_space['labels'],
        'training_source': data_meta,
        'model': task.get('model', {}),
    }
    with open(out_dir / 'run_info.json', 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2)

    meta_df.head(100).to_csv(out_dir / 'training_preview.csv', index=False)

    print(f"Saved: {out_dir / 'model.pkl'}")
    print(f"Saved: {out_dir / 'task.json'}")
    print(f"Saved: {out_dir / 'label_map.json'}")
    print(f"Saved: {out_dir / 'config.json'}")
    print(f"Saved: {out_dir / 'run_info.json'}")
    print(f"Saved: {out_dir / 'training_preview.csv'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a task-defined model')
    parser.add_argument('--task', required=True, type=str, help='Path to task JSON')
    parser.add_argument('--output-dir', default='', type=str, help='Optional override output directory')
    main(parser.parse_args())
