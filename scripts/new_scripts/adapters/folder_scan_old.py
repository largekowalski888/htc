from __future__ import annotations

from pathlib import Path
import pandas as pd


def parse_subject_and_timestamp(sample_dir: Path) -> tuple[str, str]:
    timestamp = sample_dir.name
    subject_name = sample_dir.parent.name if sample_dir.parent else sample_dir.name
    return subject_name, timestamp


def find_primary_spectrum_file(hypergui_dir: Path) -> Path | None:
    candidates = sorted(hypergui_dir.glob('spectrum_fromCSV*_masked_data.xlsx'))
    return candidates[0] if candidates else None


def scan_class_roots(class_roots: list[str | Path], output_csv: str | Path, recurse_token: str = '_hypergui_1') -> pd.DataFrame:
    rows = []
    for root_raw in class_roots:
        root = Path(root_raw)
        if not root.exists():
            raise FileNotFoundError(f'Class root does not exist: {root}')
        class_name = root.name
        for hypergui_dir in root.rglob(recurse_token):
            if not hypergui_dir.is_dir():
                continue
            sample_dir = hypergui_dir.parent
            spectrum_path = find_primary_spectrum_file(hypergui_dir)
            mask_path = hypergui_dir / 'mask.csv'
            if spectrum_path is None:
                continue
            subject_name, timestamp = parse_subject_and_timestamp(sample_dir)
            rows.append({
                'class_name': class_name,
                'class_root': str(root),
                'sample_dir': str(sample_dir),
                'subject_name': subject_name,
                'timestamp': timestamp,
                'spectrum_path': str(spectrum_path),
                'mask_path': str(mask_path) if mask_path.exists() else '',
            })

    if not rows:
        raise ValueError('No usable samples found under the provided class roots')

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
