from __future__ import annotations

import argparse
from adapters.folder_scan import scan_class_roots, expand_parent_roots


def main(args):
    if bool(args.class_roots) == bool(args.parent_roots):
        raise SystemExit('Use exactly one of --class-roots or --parent-roots')

    if args.class_roots:
        class_roots = [p.strip() for p in args.class_roots.split(';') if p.strip()]
        data_subdir = ''
    else:
        parent_roots = [p.strip() for p in args.parent_roots.split(';') if p.strip()]
        class_roots = expand_parent_roots(
            parent_roots,
            data_subdir=args.data_subdir,
            recursive=args.discover_recursive,
        )
        data_subdir = args.data_subdir

    print('Discovered class roots:')
    for p in class_roots:
        print(f'  - {p}')

    df = scan_class_roots(
        class_roots,
        args.output_csv,
        recurse_token=args.recurse_token,
        data_subdir=data_subdir,
    )
    print(f'Saved: {args.output_csv}')
    print(f'Samples found: {len(df)}')
    print(f'Classes: {sorted(df["class_name"].unique().tolist())}')
    print(df.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build a manifest from explicit class roots or from parent folders that contain many class folders'
    )
    parser.add_argument(
        '--class-roots',
        default='',
        type=str,
        help='Semicolon-separated class root directories (old behavior). Example: C:\\data\\classA;C:\\data\\classB',
    )
    parser.add_argument(
        '--parent-roots',
        default='',
        type=str,
        help='Semicolon-separated parent directories. Every child folder containing a data folder will be treated as a class root.',
    )
    parser.add_argument(
        '--data-subdir',
        default='data',
        type=str,
        help='Name of the data subfolder inside each discovered class folder (default: data)',
    )
    parser.add_argument(
        '--discover-recursive',
        action='store_true',
        help='Recursively search under each parent root for folders that contain the data subfolder',
    )
    parser.add_argument(
        '--recurse-token',
        default='_hypergui_1',
        type=str,
        help='Folder name to search for under each class/data folder (default: _hypergui_1)',
    )
    parser.add_argument('--output-csv', required=True, type=str, help='Where to save the manifest CSV')
    main(parser.parse_args())
