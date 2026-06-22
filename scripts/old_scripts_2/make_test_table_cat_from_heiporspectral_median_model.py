from pathlib import Path
import argparse
import json
import ast
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from htc import Config, LabelMapping
from htc_projects.atlas.settings_atlas import settings_atlas


def parse_vector_cell(cell):
    """Parse a spectral vector stored as a string/list/array into a 1D numpy array."""
    if isinstance(cell, np.ndarray):
        return cell.astype(np.float32).reshape(-1)
    if isinstance(cell, list):
        return np.asarray(cell, dtype=np.float32).reshape(-1)
    if isinstance(cell, tuple):
        return np.asarray(cell, dtype=np.float32).reshape(-1)
    if pd.isna(cell):
        return None
    s = str(cell).strip()
    if not s:
        return None

    # Try fast parser for strings like "[0.1 0.2 0.3]" or "[0.1, 0.2, 0.3]"
    s_inner = s.strip('[]').replace(',', ' ')
    vec = np.fromstring(s_inner, sep=' ', dtype=np.float32)
    if vec.size > 0:
        return vec

    # Fallback: literal_eval
    try:
        obj = ast.literal_eval(s)
        return np.asarray(obj, dtype=np.float32).reshape(-1)
    except Exception:
        return None


def l1_normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.sum(np.abs(vec))
    if denom == 0:
        return vec.astype(np.float32)
    out = vec / denom
    out = np.nan_to_num(out, copy=False)
    return out.astype(np.float32)


def load_heipor_table(table_path: Path, feature_column: str, mapping: LabelMapping):
    if table_path.suffix.lower() == '.feather':
        df = pd.read_feather(table_path)
    else:
        df = pd.read_csv(table_path)

    required = {'label_name', feature_column, 'subject_name', 'image_name'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'HeiPorSPECTRAL table missing required columns: {sorted(missing)}')

    rows = []
    lengths = []
    skipped = 0
    valid_names = set(mapping.label_names())

    for _, row in df.iterrows():
        label_name = str(row['label_name'])
        if label_name not in valid_names:
            skipped += 1
            continue
        vec = parse_vector_cell(row[feature_column])
        if vec is None or vec.size == 0:
            skipped += 1
            continue
        rows.append((row['image_name'], row['subject_name'], label_name, vec))
        lengths.append(vec.size)

    if not rows:
        raise ValueError('No valid HeiPorSPECTRAL training rows could be parsed')

    # Keep only the most common vector length to guarantee a consistent feature matrix
    target_len = pd.Series(lengths).mode().iloc[0]
    filtered = [(img, subj, lbl, vec) for (img, subj, lbl, vec) in rows if vec.size == target_len]
    if not filtered:
        raise ValueError('No HeiPorSPECTRAL rows left after enforcing common vector length')

    X = np.stack([vec for _, _, _, vec in filtered], axis=0)
    y_names = [lbl for _, _, lbl, _ in filtered]
    y = np.asarray([mapping.name_to_index(lbl) for lbl in y_names], dtype=np.int64)

    meta = {
        'n_total_rows': int(len(df)),
        'n_parsed_rows': int(len(rows)),
        'n_filtered_rows': int(len(filtered)),
        'feature_length': int(target_len),
        'feature_column': feature_column,
        'n_skipped': int(skipped),
    }
    return X, y, meta


def extract_cat_feature_vector(xlsx_path: Path, sheet_name: str, target_len: int, normalize: bool):
    # Read the CAT spectrum workbook. The 0_derivative sheet is expected to contain the raw masked spectrum.
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, engine='openpyxl')
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    if df.empty:
        raise ValueError(f'Empty CAT workbook sheet: {xlsx_path} [{sheet_name}]')

    # Convert to numeric where possible
    numeric = df.apply(pd.to_numeric, errors='coerce')

    values = None
    # Case 1: two-column table [wavelength, value]
    if numeric.shape[1] >= 2:
        col0 = numeric.iloc[:, 0].to_numpy()
        # choose the column with the most finite entries after the first numeric column
        best_col = None
        best_count = -1
        for j in range(1, numeric.shape[1]):
            count = np.isfinite(numeric.iloc[:, j].to_numpy()).sum()
            if count > best_count:
                best_count = count
                best_col = j
        cand = numeric.iloc[:, best_col].to_numpy()
        cand = cand[np.isfinite(cand)]
        if cand.size > 0:
            values = cand.astype(np.float32)

    # Case 2: single-column alternating wavelength/value format
    if values is None:
        flat = numeric.to_numpy().reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            raise ValueError(f'No numeric values found in CAT workbook: {xlsx_path}')
        if flat.size % 2 == 0 and np.nanmax(flat[::2]) > 100 and np.nanmin(flat[::2]) >= 400:
            values = flat[1::2].astype(np.float32)
        else:
            values = flat.astype(np.float32)

    # Align to training feature length if needed
    if values.size != target_len:
        # If lengths differ slightly, interpolate to target length
        x_old = np.linspace(0, 1, num=values.size, dtype=np.float32)
        x_new = np.linspace(0, 1, num=target_len, dtype=np.float32)
        values = np.interp(x_new, x_old, values).astype(np.float32)

    # Normalize to match training representation if using normalized spectra
    if normalize:
        values = l1_normalize(values)
    else:
        values = values.astype(np.float32)

    return values



def main(args):
    heipor_table = Path(args.heipor_table)
    cat_manifest = Path(args.cat_manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Config({
        'label_mapping': settings_atlas.label_mapping
    })
    mapping = LabelMapping.from_config(config)
    n_classes = len(mapping)
    kidney_index = mapping.name_to_index('kidney')

    X_train, y_train, train_meta = load_heipor_table(heipor_table, args.feature_column, mapping)
    target_len = X_train.shape[1]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight='balanced_subsample',
    )
    print(f'Training on HeiPorSPECTRAL rows: {X_train.shape[0]} samples x {X_train.shape[1]} features')
    model.fit(X_train, y_train)

    df_cat = pd.read_csv(cat_manifest)
    required = {'sample_dir', 'subject_name', 'timestamp', 'condition', 'mask_path', 'spectrum_path'}
    missing = required - set(df_cat.columns)
    if missing:
        raise SystemExit(f'CAT manifest missing required columns: {sorted(missing)}')

    if args.limit > 0:
        df_cat = df_cat.head(args.limit).copy()
    if args.keep_conditions:
        keep = {c.strip() for c in args.keep_conditions.split(',') if c.strip()}
        df_cat = df_cat[df_cat['condition'].isin(keep)].copy()
    if df_cat.empty:
        raise SystemExit('No CAT rows left after filtering. Check manifest and --keep-conditions.')

    rows_by_subject = {}
    qc_rows = []

    for _, row in df_cat.iterrows():
        subject_name = row['subject_name']
        timestamp = row['timestamp']
        xlsx_path = Path(row['spectrum_path'])
        try:
            normalize_cat = (args.feature_column == "median_normalized_spectrum")
            x = extract_cat_feature_vector(xlsx_path, args.cat_sheet, target_len, normalize=normalize_cat)
            y_true = np.array([kidney_index], dtype=np.int64)
            y_pred = model.predict(x.reshape(1, -1)).astype(np.int64)
            proba = None
            pred_conf = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(x.reshape(1, -1))[0]
                pred_conf = float(np.max(proba))
            cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
            acc = accuracy_score(y_true, y_pred)
            pred_idx = int(y_pred[0])
            pred_name = mapping.index_to_name(pred_idx)

            rows_by_subject.setdefault(subject_name, []).append({
                'timestamp': timestamp,
                'condition': row['condition'],
                'sample_dir': row['sample_dir'],
                'spectrum_path': row['spectrum_path'],
                'confusion_matrix': cm,
                'accuracy': float(acc),
            })

            qc_rows.append({
                'subject_name': subject_name,
                'timestamp': timestamp,
                'condition': row['condition'],
                'sample_dir': row['sample_dir'],
                'spectrum_path': row['spectrum_path'],
                'true_label_name': 'kidney',
                'true_label_index': int(kidney_index),
                'pred_label_name': pred_name,
                'pred_label_index': pred_idx,
                'pred_confidence': pred_conf,
                'accuracy': float(acc),
                'status': 'ok',
            })

        except Exception as e:
                print(f"FAILED CAT ROW -> subject={subject_name}, timestamp={timestamp}")
                print(f"spectrum_path={xlsx_path}")
                print(f"ERROR: {e}")
                qc_rows.append({
                    'subject_name': subject_name,
                    'timestamp': timestamp,
                    'condition': row['condition'],
                    'sample_dir': row['sample_dir'],
                    'spectrum_path': row['spectrum_path'],
                    'true_label_name': 'kidney',
                    'true_label_index': int(kidney_index),
                    'pred_label_name': None,
                    'pred_label_index': None,
                    'pred_confidence': None,
                    'accuracy': None,
                    'status': f'error: {e}',
                })
                break


    if not rows_by_subject:
        raise SystemExit('No valid CAT rows processed. Check spectrum paths and workbook format.')

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
    df_test.to_pickle(out_dir / 'test_table.pkl.xz')
    config.save_config(out_dir / 'config.json')

    df_qc = pd.DataFrame(qc_rows)
    df_qc.to_csv(out_dir / 'qc_report.csv', index=False)

    with open(out_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)

    run_info = {
        'heipor_table': str(heipor_table),
        'cat_manifest': str(cat_manifest),
        'n_cat_rows_after_filtering': int(len(df_cat)),
        'n_subject_rows': int(len(df_test)),
        'mode': 'transfer model trained on HeiPorSPECTRAL median spectra, applied to CAT spectra',
        'feature_column': args.feature_column,
        'cat_sheet': args.cat_sheet,
        'feature_length': int(target_len),
        'kidney_index': int(kidney_index),
        'label_names': mapping.label_names(),
        'train_meta': train_meta,
        'n_estimators': int(args.n_estimators),
        'max_depth': (None if args.max_depth is None else int(args.max_depth)),
        'random_state': int(args.random_state),
    }
    with open(out_dir / 'run_info.json', 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2)

    print(f"Saved: {out_dir / 'test_table.pkl.xz'}")
    print(f"Saved: {out_dir / 'config.json'}")
    print(f"Saved: {out_dir / 'qc_report.csv'}")
    print(f"Saved: {out_dir / 'run_info.json'}")
    print(f"Saved: {out_dir / 'model.pkl'}")
    print(f"Subjects processed successfully: {len(df_test)}")
    print(f"Training rows used: {train_meta['n_filtered_rows']}")
    print(f"Feature length: {target_len}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on HeiPorSPECTRAL median spectra and predict CAT spectra in full 20-class label space')
    parser.add_argument('--heipor-table', type=str, required=True, help='Path to HeiPorSPECTRAL median spectra table (.csv or .feather)')
    parser.add_argument('--cat-manifest', type=str, default=r'C:\DKFZ\htc\scripts\cat_manifest.csv')
    parser.add_argument('--output-dir', type=str, default=r'C:\DKFZ\outputs\cat_from_heipor_median_model')
    parser.add_argument('--feature-column', type=str, default='median_normalized_spectrum', help='HeiPorSPECTRAL feature column to train on')
    parser.add_argument('--cat-sheet', type=str, default='0_derivative', help='Sheet in CAT workbook to use as feature source')
    parser.add_argument('--limit', type=int, default=0, help='Optional: only use first N CAT manifest rows')
    parser.add_argument('--keep-conditions', type=str, default='', help='Optional: comma-separated conditions to keep (e.g. baseline,deox)')
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--random-state', type=int, default=42)
    main(parser.parse_args())
