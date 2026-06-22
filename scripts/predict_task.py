from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle

import numpy as np
import pandas as pd

from htc import Config, LabelMapping
from data_adapters import load_cat_manifest, extract_cat_feature_vector


def main(args):
    run_dir = Path(args.run_dir)
    task = json.loads((run_dir / 'task.json').read_text(encoding='utf-8'))
    config = Config(run_dir / 'config.json')
    mapping = LabelMapping.from_config(config)

    with open(run_dir / 'model.pkl', 'rb') as f:
        model = pickle.load(f)

    inf_cfg = task['inference_source']
    if inf_cfg['adapter'] != 'cat_manifest':
        raise ValueError('predict_task currently supports only cat_manifest inference')

    keep_conditions = None
    if args.keep_conditions:
        keep_conditions = [c.strip() for c in args.keep_conditions.split(',') if c.strip()]
    elif inf_cfg.get('keep_conditions'):
        keep_conditions = list(inf_cfg['keep_conditions'])

    df_cat = load_cat_manifest(inf_cfg['path'], keep_conditions=keep_conditions, limit=int(args.limit or 0))
    target_len = None
    # infer expected feature length from model if possible
    if hasattr(model, 'n_features_in_'):
        target_len = int(model.n_features_in_)
    else:
        # fallback: read run_info.json
        run_info = json.loads((run_dir / 'run_info.json').read_text(encoding='utf-8'))
        target_len = int(run_info['feature_length'])

    normalize = bool(inf_cfg['feature'].get('normalize', True))
    sheet = inf_cfg['feature'].get('sheet', '0_derivative')

    pred_rows = []
    for _, row in df_cat.iterrows():
        x = extract_cat_feature_vector(row['spectrum_path'], sheet_name=sheet, target_len=target_len, normalize=normalize)
        pred_idx = int(model.predict(x.reshape(1, -1))[0])
        pred_name = mapping.label_names()[pred_idx]
        pred_conf = None
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(x.reshape(1, -1))[0]
            pred_conf = float(np.max(p))
        pred_rows.append({
            'subject_name': row['subject_name'],
            'timestamp': row['timestamp'],
            'condition': row['condition'],
            'sample_dir': row['sample_dir'],
            'spectrum_path': row['spectrum_path'],
            'pred_label_index': pred_idx,
            'pred_label_name': pred_name,
            'pred_confidence': pred_conf,
        })

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(run_dir / 'predictions.csv', index=False)
    print(f"Saved: {run_dir / 'predictions.csv'}")
    print(f"Predictions rows: {len(pred_df)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference for a trained task')
    parser.add_argument('--run-dir', required=True, type=str, help='Directory produced by train_task.py')
    parser.add_argument('--limit', default=0, type=int, help='Optional limit on inference rows')
    parser.add_argument('--keep-conditions', default='', type=str, help='Optional comma-separated condition filter override')
    main(parser.parse_args())
