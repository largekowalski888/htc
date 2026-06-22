from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score


def main(args):
    manifest_path = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    required = {"sample_dir", "subject_name", "timestamp", "condition", "mask_path", "spectrum_path"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest missing required columns: {sorted(missing)}")

    if args.limit > 0:
        df = df.head(args.limit).copy()

    # Optionally filter only certain CAT conditions if the manifest mixes multiple roots later
    if args.keep_conditions:
        keep = {c.strip() for c in args.keep_conditions.split(',') if c.strip()}
        df = df[df['condition'].isin(keep)].copy()

    if df.empty:
        raise SystemExit("No rows left after filtering. Check manifest and --keep-conditions.")

    # In this scaffold all CAT rows are treated as positive (kidney) samples.
    # We create one confusion matrix per subject aggregated over all timestamps/images.
    rows_by_subject = {}
    qc_rows = []

    for _, row in df.iterrows():
        subject_name = row['subject_name']
        timestamp = row['timestamp']

        # Sample-level truth/prediction for the scaffold:
        # all CAT samples are assumed kidney, and placeholder prediction is also kidney.
        y_true = np.array([1], dtype=np.int64)
        y_pred = np.array([1], dtype=np.int64)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc = accuracy_score(y_true, y_pred)

        rows_by_subject.setdefault(subject_name, []).append({
            'timestamp': timestamp,
            'condition': row['condition'],
            'sample_dir': row['sample_dir'],
            'mask_path': row['mask_path'],
            'spectrum_path': row['spectrum_path'],
            'confusion_matrix': cm,
            'accuracy': float(acc),
        })

        qc_rows.append({
            'subject_name': subject_name,
            'timestamp': timestamp,
            'condition': row['condition'],
            'sample_dir': row['sample_dir'],
            'mask_path': row['mask_path'],
            'spectrum_path': row['spectrum_path'],
            'y_true': int(y_true[0]),
            'y_pred': int(y_pred[0]),
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
    df_test.to_pickle(out_dir / 'test_table.pkl.xz')

    config = {
        'label_mapping': {
            'not_kidney': 0,
            'kidney': 1
        }
    }
    with open(out_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    pd.DataFrame(qc_rows).to_csv(out_dir / 'qc_report.csv', index=False)

    run_info = {
        'manifest': str(manifest_path),
        'n_manifest_rows_after_filtering': int(len(df)),
        'n_subject_rows': int(len(df_test)),
        'mode': 'cat kidney-positive scaffold',
        'note': 'All CAT samples assumed kidney; placeholder predictions also kidney',
    }
    with open(out_dir / 'run_info.json', 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2)

    print(f"Saved: {out_dir / 'test_table.pkl.xz'}")
    print(f"Saved: {out_dir / 'config.json'}")
    print(f"Saved: {out_dir / 'qc_report.csv'}")
    print(f"Saved: {out_dir / 'run_info.json'}")
    print(f"Subjects processed successfully: {len(df_test)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create CAT kidney-positive external-validation scaffold from manifest')
    parser.add_argument('--manifest', type=str, default=r'C:\DKFZ\scripts\cat_manifest.csv')
    parser.add_argument('--output-dir', type=str, default=r'C:\DKFZ\outputs\cat_kidney_positive_scaffold')
    parser.add_argument('--limit', type=int, default=0, help='Optional: only use first N manifest rows')
    parser.add_argument('--keep-conditions', type=str, default='', help='Optional: comma-separated conditions to keep (e.g. baseline,deox)')
    main(parser.parse_args())
