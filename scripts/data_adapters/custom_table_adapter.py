from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .heipor_adapter import parse_vector_cell


def load_custom_table(table_path: str | Path, feature_column: str, label_column: str,
                      group_column: str | None = None, normalize_features: bool = False) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    table_path = Path(table_path)
    if table_path.suffix.lower() == '.feather':
        df = pd.read_feather(table_path)
    else:
        df = pd.read_csv(table_path)

    required = {feature_column, label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Custom table missing required columns: {sorted(missing)}')

    vecs = []
    labels = []
    keep_rows = []
    lengths = []
    for _, row in df.iterrows():
        vec = parse_vector_cell(row[feature_column])
        if vec is None or vec.size == 0:
            continue
        vecs.append(vec)
        labels.append(row[label_column])
        keep_rows.append(row.to_dict())
        lengths.append(vec.size)

    if not vecs:
        raise ValueError('No valid rows found in custom table')

    target_len = int(pd.Series(lengths).mode().iloc[0])
    zipped = [(v, y, m) for v, y, m in zip(vecs, labels, keep_rows) if v.size == target_len]
    X = np.stack([v for v, _, _ in zipped], axis=0).astype(np.float32)
    if normalize_features:
        denom = np.sum(np.abs(X), axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        X = np.nan_to_num(X / denom, copy=False).astype(np.float32)
    y_raw = [y for _, y, _ in zipped]
    meta_df = pd.DataFrame([m for _, _, m in zipped])

    label_names = sorted(pd.Series(y_raw).astype(str).unique().tolist())
    name_to_index = {name: i for i, name in enumerate(label_names)}
    y = np.asarray([name_to_index[str(v)] for v in y_raw], dtype=np.int64)

    metadata = {
        'n_total_rows': int(len(df)),
        'n_filtered_rows': int(len(zipped)),
        'feature_length': int(target_len),
        'feature_column': feature_column,
        'label_column': label_column,
        'normalize_features': bool(normalize_features),
        'labels': label_names,
        'group_column': group_column,
    }
    return X, y, meta_df, metadata
