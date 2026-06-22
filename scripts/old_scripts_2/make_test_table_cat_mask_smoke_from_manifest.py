
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score


def load_mask(mask_path: Path) -> np.ndarray:
    arr = pd.read_csv(mask_path, header=None).to_numpy(dtype=np.int64)
    uniq = np.unique(arr)
    if not set(uniq.tolist()).issubset({0, 1}):
        raise ValueError(f"Mask is not binary: {mask_path} unique={uniq[:10]}")
    return arr


def inject_noise(y_true: np.ndarray, noise_fraction: float, rng: np.random.Generator) -> np.ndarray:
    y_pred = y_true.copy()
    if noise_fraction <= 0:
        return y_pred
    n_flip = max(1, int(round(len(y_pred) * noise_fraction)))
    flip_idx = rng.choice(len(y_pred), size=n_flip, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    return y_pred


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

    rng = np.random.default_rng(args.random_state)

    # Store per-subject results here
    rows_by_subject = {}
    qc_rows = []

    for _, row in df.iterrows():
        mask_path = Path(row["mask_path"])
        subject_name = row["subject_name"]
        timestamp = row["timestamp"]

        try:
            mask = load_mask(mask_path)
            y_true = mask.reshape(-1)
            y_pred = inject_noise(y_true, args.noise_fraction, rng)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            acc = accuracy_score(y_true, y_pred)

            rows_by_subject.setdefault(subject_name, []).append({
                "timestamp": timestamp,
                "condition": row["condition"],
                "confusion_matrix": cm,
                "accuracy": float(acc),
            })

            qc_rows.append({
                "subject_name": subject_name,
                "timestamp": timestamp,
                "condition": row["condition"],
                "mask_path": str(mask_path),
                "mask_shape": tuple(mask.shape),
                "n_positive": int(mask.sum()),
                "n_total": int(mask.size),
                "accuracy": float(acc),
                "status": "ok",
            })

        except Exception as e:
            qc_rows.append({
                "subject_name": subject_name,
                "timestamp": timestamp,
                "condition": row["condition"],
                "mask_path": str(mask_path),
                "mask_shape": None,
                "n_positive": None,
                "n_total": None,
                "accuracy": None,
                "status": f"error: {e}",
            })

    if not rows_by_subject:
        raise SystemExit("No valid rows processed. Check manifest paths and mask files.")

    # ---- AGGREGATE TO ONE ROW PER SUBJECT ----
    subject_rows = []
    for subject_name, items in rows_by_subject.items():
        cms = np.stack([it["confusion_matrix"] for it in items], axis=0)
        subject_cm = cms.sum(axis=0)
        mean_acc = float(np.mean([it["accuracy"] for it in items]))

        subject_rows.append({
            "subject_name": subject_name,
            "image_names": [it["timestamp"] for it in items],
            "n_images": len(items),
            "condition": items[0]["condition"],
            "confusion_matrix": subject_cm,
            "accuracy": mean_acc,
        })

    df_test = pd.DataFrame(subject_rows)
    df_test.to_pickle(out_dir / "test_table.pkl.xz")

    config = {
        "label_mapping": {
            "background": 0,
            "roi": 1
        }
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    df_qc = pd.DataFrame(qc_rows)
    df_qc.to_csv(out_dir / "qc_report.csv", index=False)

    run_info = {
        "manifest": str(manifest_path),
        "n_manifest_rows": int(len(df)),
        "n_subject_rows": int(len(df_test)),
        "noise_fraction": float(args.noise_fraction),
        "random_state": int(args.random_state),
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print(f"Saved: {out_dir / 'test_table.pkl.xz'}")
    print(f"Saved: {out_dir / 'config.json'}")
    print(f"Saved: {out_dir / 'qc_report.csv'}")
    print(f"Saved: {out_dir / 'run_info.json'}")
    print(f"Subjects processed successfully: {len(df_test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create CAT binary smoke-test table from manifest + binary masks")
    parser.add_argument("--manifest", type=str, default=r"cat_manifest.csv")
    parser.add_argument("--output-dir", type=str, default=r"C:\DKFZ\outputs\cat_mask_smoke_run")
    parser.add_argument("--noise-fraction", type=float, default=0.0, help="0.0 = perfect predictions; e.g. 0.05 for 5 percent flips")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Optional: only use first N manifest rows for quick testing")
    main(parser.parse_args())
