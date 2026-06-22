from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import balanced_accuracy_score

from htc import Config
from adapters.spectrum_adapter import extract_spectrum_vector


def build_label_space(df: pd.DataFrame) -> tuple[list[str], dict[str, int], Config]:
    labels = sorted(df['class_name'].astype(str).unique().tolist())
    name_to_index = {name: i for i, name in enumerate(labels)}
    config = Config({'label_mapping': name_to_index})
    return labels, name_to_index, config


def split_manifest_3way(df: pd.DataFrame, val_size: float, test_size: float, random_state: int, group_by_subject: bool):
    if val_size + test_size >= 1.0:
        raise ValueError('val_size + test_size must be < 1.0')

    # 1) full -> train_val + test
    if group_by_subject and df['subject_name'].nunique() >= 3:
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss_test.split(df, y=df['class_name'], groups=df['subject_name']))
    else:
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(sss_test.split(df, df['class_name']))

    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # 2) train_val -> train + val
    val_relative = val_size / (1.0 - test_size)
    if group_by_subject and train_val_df['subject_name'].nunique() >= 2:
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(train_val_df, y=train_val_df['class_name'], groups=train_val_df['subject_name']))
    else:
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=random_state)
        train_idx, val_idx = next(sss_val.split(train_val_df, train_val_df['class_name']))

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()
    return train_df, val_df, test_df


def load_feature_matrix(df: pd.DataFrame, sheet_name: str, normalize: bool, target_len: int | None = None):
    X = []
    ok_rows = []
    errors = []
    inferred_len = target_len
    for _, row in df.iterrows():
        try:
            vec = extract_spectrum_vector(row['spectrum_path'], sheet_name=sheet_name, target_len=inferred_len, normalize=normalize)
            if inferred_len is None:
                inferred_len = vec.size
            X.append(vec)
            ok_rows.append(row)
        except Exception as e:
            err = row.to_dict()
            err['status'] = f'error: {e}'
            errors.append(err)
    if not X:
        raise ValueError('No usable feature vectors could be extracted')
    return np.stack(X, axis=0), pd.DataFrame(ok_rows), pd.DataFrame(errors), inferred_len


def aggregate_subject_rows(df_eval: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> pd.DataFrame:
    rows_by_subject = {}
    for i, row in df_eval.reset_index(drop=True).iterrows():
        subject = row['subject_name']
        cm = confusion_matrix([int(y_true[i])], [int(y_pred[i])], labels=list(range(n_classes)))
        acc = float(accuracy_score([int(y_true[i])], [int(y_pred[i])]))
        rows_by_subject.setdefault(subject, []).append({
            'timestamp': row['timestamp'],
            'class_name': row['class_name'],
            'confusion_matrix': cm,
            'accuracy': acc,
        })

    rows = []
    for subject, items in rows_by_subject.items():
        cms = np.stack([it['confusion_matrix'] for it in items], axis=0)
        rows.append({
            'subject_name': subject,
            'image_names': [it['timestamp'] for it in items],
            'n_images': len(items),
            'condition': items[0]['class_name'],
            'confusion_matrix': cms.sum(axis=0),
            'accuracy': float(np.mean([it['accuracy'] for it in items])),
        })
    return pd.DataFrame(rows)


def main(args):
    manifest_path = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    required = {'class_name', 'subject_name', 'timestamp', 'sample_dir', 'spectrum_path'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'Manifest missing columns: {sorted(missing)}')

    labels, name_to_index, config = build_label_space(df)
    df['label_index'] = df['class_name'].map(name_to_index)

    train_df, val_df, test_df = split_manifest_3way(
        df,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        group_by_subject=not args.no_group_split,
    )

    X_train, ok_train_df, train_err_df, target_len = load_feature_matrix(train_df, args.sheet, args.normalize, target_len=None)
    X_val, ok_val_df, val_err_df, _ = load_feature_matrix(val_df, args.sheet, args.normalize, target_len=target_len)
    X_test, ok_test_df, test_err_df, _ = load_feature_matrix(test_df, args.sheet, args.normalize, target_len=target_len)

    y_train = ok_train_df['label_index'].to_numpy(dtype=np.int64)
    y_val = ok_val_df['label_index'].to_numpy(dtype=np.int64)
    y_test = ok_test_df['label_index'].to_numpy(dtype=np.int64)

    candidates = [int(x) for x in args.n_estimators_grid.split(',') if x.strip()]
    depth_candidates = []
    for x in args.max_depth_grid.split(','):
        x = x.strip()
        if not x:
            continue
        depth_candidates.append(None if x.lower() == 'none' else int(x))

    best_model = None
    best_cfg = None
    best_val_acc = -1.0
    val_results = []

    for n_estimators in candidates:
        for max_depth in depth_candidates:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=args.random_state,
                n_jobs=-1,
                class_weight='balanced_subsample',
            )
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val).astype(np.int64)
            val_acc = balanced_accuracy_score(y_val, y_val_pred)
            val_results.append({'n_estimators': n_estimators, 'max_depth': max_depth, 'val_acc': val_acc})
            print(f'candidate n_estimators={n_estimators} max_depth={max_depth} -> val_acc={val_acc:.4f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                best_cfg = {'n_estimators': n_estimators, 'max_depth': max_depth}

    assert best_model is not None
    print(f"Best RF config: {best_cfg} with val_acc={best_val_acc:.4f}")

    y_test_pred = best_model.predict(X_test).astype(np.int64)
    test_acc = float((y_test_pred == y_test).mean())

    df_subject = aggregate_subject_rows(ok_test_df, y_test, y_test_pred, n_classes=len(labels))
    df_subject.to_pickle(out_dir / 'test_table.pkl.xz')
    config.save_config(out_dir / 'config.json')

    pred_df = ok_test_df.copy()
    pred_df['true_label_index'] = y_test
    pred_df['pred_label_index'] = y_test_pred
    pred_df['true_label_name'] = [labels[i] for i in y_test]
    pred_df['pred_label_name'] = [labels[i] for i in y_test_pred]
    pred_df.to_csv(out_dir / 'predictions.csv', index=False)

    qc_df = pd.concat([train_err_df, val_err_df, test_err_df], ignore_index=True) if (not train_err_df.empty or not val_err_df.empty or not test_err_df.empty) else pd.DataFrame(columns=list(df.columns)+['status'])
    qc_df.to_csv(out_dir / 'qc_report.csv', index=False)

    pd.DataFrame(val_results).to_csv(out_dir / 'validation_results.csv', index=False)

    with open(out_dir / 'model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open(out_dir / 'label_map.json', 'w', encoding='utf-8') as f:
        json.dump({'labels': labels, 'name_to_index': name_to_index}, f, indent=2)
    with open(out_dir / 'run_info.json', 'w', encoding='utf-8') as f:
        json.dump({
            'mode': 'folder->manifest->RF (train/val/test)',
            'manifest': str(manifest_path),
            'n_total': int(len(df)),
            'n_train_manifest_rows': int(len(train_df)),
            'n_val_manifest_rows': int(len(val_df)),
            'n_test_manifest_rows': int(len(test_df)),
            'n_train_used': int(len(ok_train_df)),
            'n_val_used': int(len(ok_val_df)),
            'n_test_used': int(len(ok_test_df)),
            'feature_length': int(target_len),
            'sheet': args.sheet,
            'normalize': bool(args.normalize),
            'labels': labels,
            'val_size': float(args.val_size),
            'test_size': float(args.test_size),
            'group_split': bool(not args.no_group_split),
            'best_model': best_cfg,
            'best_val_accuracy': float(best_val_acc),
            'mean_test_accuracy': float(test_acc),
        }, f, indent=2)

    ok_train_df.to_csv(out_dir / 'train_manifest.csv', index=False)
    ok_val_df.to_csv(out_dir / 'val_manifest.csv', index=False)
    ok_test_df.to_csv(out_dir / 'test_manifest.csv', index=False)

    print(f'Saved: {out_dir / "model.pkl"}')
    print(f'Saved: {out_dir / "config.json"}')
    print(f'Saved: {out_dir / "test_table.pkl.xz"}')
    print(f'Saved: {out_dir / "predictions.csv"}')
    print(f'Saved: {out_dir / "qc_report.csv"}')
    print(f'Saved: {out_dir / "validation_results.csv"}')
    print(f'Saved: {out_dir / "run_info.json"}')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Mean test accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train a Random Forest from a folder-derived manifest with train/val/test split and write HTC-style outputs')
    p.add_argument('--manifest', required=True, type=str)
    p.add_argument('--output-dir', required=True, type=str)
    p.add_argument('--sheet', default='0_derivative', type=str)
    p.add_argument('--normalize', action='store_true', help='L1-normalize extracted workbook vectors')
    p.add_argument('--val-size', default=0.15, type=float, help='Validation fraction of total dataset')
    p.add_argument('--test-size', default=0.15, type=float, help='Test fraction of total dataset')
    p.add_argument('--random-state', default=42, type=int)
    p.add_argument('--n-estimators-grid', default='100,300,500', type=str, help='Comma-separated RF n_estimators candidates')
    p.add_argument('--max-depth-grid', default='None,10,20', type=str, help='Comma-separated RF max_depth candidates (use None)')
    p.add_argument('--no-group-split', action='store_true', help='Disable subject-level split and use stratified sample split')
    main(p.parse_args())
