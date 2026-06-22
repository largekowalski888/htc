from __future__ import annotations

import argparse
from adapters.folder_scan import scan_class_roots


def main(args):
    class_roots = [p.strip() for p in args.class_roots.split(';') if p.strip()]
    df = scan_class_roots(class_roots, args.output_csv)
    print(f'Saved: {args.output_csv}')
    print(f'Samples found: {len(df)}')
    print(f'Classes: {sorted(df["class_name"].unique().tolist())}')
    print(df.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a manifest from folder-based class roots')
    parser.add_argument('--class-roots', required=True, type=str, help='Semicolon-separated class root directories')
    parser.add_argument('--output-csv', required=True, type=str, help='Where to save the manifest CSV')
    main(parser.parse_args())
