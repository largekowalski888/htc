from __future__ import annotations

from pathlib import Path
import ast
import json
from typing import Callable

import numpy as np
import pandas as pd

from htc import Config, LabelMapping
from htc_projects.atlas.settings_atlas import settings_atlas


def parse_vector_cell(cell) -> np.ndarray | None:
    if isinstance(cell, np.ndarray):
        vec = cell.astype(np.float32).reshape(-1)
        return vec if vec.size > 0 else None
    if isinstance(cell, (list, tuple)):
        vec = np.asarray(cell, dtype=np.float32).reshape(-1)
        return vec if vec.size > 0 else None
    if pd.isna(cell):
        return None

    s = str(cell).strip()
    if not s:
        return None

    s_inner = s.strip('[]').replace(',', ' ')
    vec = np.fromstring(s_inner, sep=' ', dtype=np.float32)
    if vec.size > 0:
        return vec

    try:
        obj = ast.literal_eval(s)
        vec = np.asarray(obj, dtype=np.float32).reshape(-1)
        return vec if vec.size > 0 else None
    except Exception:
        return None


def _l1_normalize_batch(X: np.ndarray) -> np.ndarray:
    denom = np.sum(np.abs(X), axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    X = X / denom
    X = np.nan_to_num(X, copy=False)
    return X.astype(np.float32)


def build_heipor_label_space(task: dict) -> dict:
    mode = task.get('label_space', {}).get('source', 'heipor_atlas_mapping')
    if mode == 'heipor_atlas_mapping':
        config = Config({'label_mapping': settings_atlas.label_mapping})
        mapping = LabelMapping.from_config(config)
        labels = mapping.label_names()
        name_to_index = {name: mapping.name_to_index(name) for name in labels}
        return {
            'labels': labels,
            'name_to_index': name_to_index,
            'config': config,
        }
    elif mode == 'explicit_labels':
        labels = task['label_space']['labels']
        name_to_index = {name: i for i, name in enumerate(labels)}
        config = Config({'label_mapping': name_to_index})
        return {
            'labels': labels,
            'name_to_index': name_to_index,
            'config': config,
        }
    else:
        raise ValueError(f'Unsupported label_space source: {mode}')


def _label_transform_factory(transform_name: str | None) -> Callable[[str], str | None]:
    if transform_name in (None, '', 'identity'):
        return lambda label: label
    if transform_name == 'kidney_vs_rest':
        return lambda label: 'kidney' if label == 'kidney' else 'not_kidney'
    raise ValueError(f'Unsupported label transform: {transform_name}')


def load_heipor_table(table_path: str | Path, feature_column: str, label_column: str, label_space: dict,
                      label_transform: str | None = None, normalize_features: bool | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    table_path = Path(table_path)
    if table_path.suffix.lower() == '.feather':
        df = pd.read_feather(table_path)
    else:
        df = pd.read_csv(table_path)

    required = {feature_column, label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'HeiPorSPECTRAL table missing required columns: {sorted(missing)}')

    transform = _label_transform_factory(label_transform)
    rows = []
    lengths = []
    skipped = 0
    valid_names = set(label_space['labels'])

    for _, row in df.iterrows():
        raw_label = str(row[label_column])
        mapped_label = transform(raw_label)
        if mapped_label is None or mapped_label not in valid_names:
            skipped += 1
            continue
        vec = parse_vector_cell(row[feature_column])
        if vec is None or vec.size == 0:
            skipped += 1
            continue
        rows.append((mapped_label, vec, row.to_dict()))
        lengths.append(vec.size)

    if not rows:
        raise ValueError('No valid HeiPorSPECTRAL rows could be parsed')

    target_len = int(pd.Series(lengths).mode().iloc[0])
    filtered = [(lbl, vec, meta) for (lbl, vec, meta) in rows if vec.size == target_len]
    if not filtered:
        raise ValueError('No training rows left after enforcing common vector length')

    X = np.stack([vec for _, vec, _ in filtered], axis=0)
    if normalize_features is True:
        X = _l1_normalize_batch(X)
    y = np.asarray([label_space['name_to_index'][lbl] for lbl, _, _ in filtered], dtype=np.int64)
    meta_df = pd.DataFrame([meta for _, _, meta in filtered])

    metadata = {
        'n_total_rows': int(len(df)),
        'n_parsed_rows': int(len(rows)),
        'n_filtered_rows': int(len(filtered)),
        'feature_length': int(target_len),
        'feature_column': feature_column,
        'label_column': label_column,
        'label_transform': label_transform or 'identity',
        'normalize_features': bool(normalize_features),
        'n_skipped': int(skipped),
    }
    return X, y, meta_df, metadata
