from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score

from htc import Config
from adapters.spectrum_adapter import extract_spectrum_vector

from tabpfn import TabPFNClassifier


def build_label_space(df: pd.DataFrame):
    labels = sorted(df["class_name"].astype(str).unique().tolist())
    name_to_index = {name: i for i, name in enumerate(labels)}
    config = Config({"label_mapping": name_to_index})
    return labels, name_to_index, config


def split_manifest(df: pd.DataFrame, test_size: float, random_state: int, group_by_subject: bool):
    if group_by_subject and df["subject_name"].nunique() >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, y=df["class_name"], groups=df["subject_name"]))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(df, df["class_name"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def load_feature_matrix(df: pd.DataFrame, sheet_name: str, normalize: bool, target_len: int | None = None):
    X = []
    ok_rows = []
    errors = []
    inferred_len = target_len

    for _, row in df.iterrows():
        try:
            vec = extract_spectrum_vector(
                row["spectrum_path"],
                sheet_name=sheet_name,
                target_len=inferred_len,
                normalize=normalize,
            )
            if inferred_len is None:
                inferred_len = vec.size
            X.append(vec)
            ok_rows.append(row)
        except Exception as e:
            err = row.to_dict()
            err["status"] = f"error: {e}"
            errors.append(err)

    if not X:
        raise ValueError("No usable feature vectors could be extracted")

    X = np.stack(X, axis=0)
    ok_df = pd.DataFrame(ok_rows)
    err_df = pd.DataFrame(errors)
    return X, ok_df, err_df, inferred_len


def aggregate_subject_rows(df_eval: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    rows_by_subject = {}

    for i, row in df_eval.reset_index(drop=True).iterrows():
        subject = row["subject_name"]
        cm = confusion_matrix([int(y_true[i])], [int(y_pred[i])], labels=list(range(n_classes)))
        acc = float(accuracy_score([int(y_true[i])], [int(y_pred[i])]))

        rows_by_subject.setdefault(subject, []).append(
            {
                "timestamp": row["timestamp"],
                "class_name": row["class_name"],
                "confusion_matrix": cm,
                "accuracy": acc,
            }
        )

    rows = []
    for subject, items in rows_by_subject.items():
        cms = np.stack([it["confusion_matrix"] for it in items], axis=0)
        rows.append(
            {
                "subject_name": subject,
                "image_names": [it["timestamp"] for it in items],
                "n_images": len(items),
                "condition": items[0]["class_name"],
                "confusion_matrix": cms.sum(axis=0),
                "accuracy": float(np.mean([it["accuracy"] for it in items])),
            }
        )
    return pd.DataFrame(rows)


def main(args):
    manifest_path = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    required = {"class_name", "subject_name", "timestamp", "sample_dir", "spectrum_path"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")

    labels, name_to_index, config = build_label_space(df)
    df["label_index"] = df["class_name"].map(name_to_index)

    train_df, test_df = split_manifest(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        group_by_subject=not args.no_group_split,
    )

    X_train, ok_train_df, train_err_df, target_len = load_feature_matrix(
        train_df, args.sheet, args.normalize, target_len=None
    )
    X_test, ok_test_df, test_err_df, _ = load_feature_matrix(
        test_df, args.sheet, args.normalize, target_len=target_len
    )

    y_train = ok_train_df["label_index"].to_numpy(dtype=np.int64)
    y_test = ok_test_df["label_index"].to_numpy(dtype=np.int64)

    print(f"Training TabPFN on {X_train.shape[0]} rows x {X_train.shape[1]} features")

    model = TabPFNClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test).astype(np.int64)

    pred_conf = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)
            pred_conf = probs.max(axis=1)
        except Exception:
            pred_conf = None

    df_subject = aggregate_subject_rows(ok_test_df, y_test, y_pred, n_classes=len(labels))
    df_subject.to_pickle(out_dir / "test_table.pkl.xz")
    config.save_config(out_dir / "config.json")

    pred_df = ok_test_df.copy()
    pred_df["true_label_index"] = y_test
    pred_df["pred_label_index"] = y_pred
    pred_df["true_label_name"] = [labels[i] for i in y_test]
    pred_df["pred_label_name"] = [labels[i] for i in y_pred]
    if pred_conf is not None:
        pred_df["pred_confidence"] = pred_conf
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    qc_df = (
        pd.concat([train_err_df, test_err_df], ignore_index=True)
        if (not train_err_df.empty or not test_err_df.empty)
        else pd.DataFrame(columns=list(df.columns) + ["status"])
    )
    qc_df.to_csv(out_dir / "qc_report.csv", index=False)

    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "name_to_index": name_to_index}, f, indent=2)

    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "folder->manifest->TabPFN",
                "manifest": str(manifest_path),
                "n_total": int(len(df)),
                "n_train_manifest_rows": int(len(train_df)),
                "n_test_manifest_rows": int(len(test_df)),
                "n_train_used": int(len(ok_train_df)),
                "n_test_used": int(len(ok_test_df)),
                "feature_length": int(target_len),
                "sheet": args.sheet,
                "normalize": bool(args.normalize),
                "labels": labels,
                "test_size": float(args.test_size),
                "group_split": bool(not args.no_group_split),
                "mean_sample_accuracy": float((y_test == y_pred).mean()),
            },
            f,
            indent=2,
        )

    ok_train_df.to_csv(out_dir / "train_manifest.csv", index=False)
    ok_test_df.to_csv(out_dir / "test_manifest.csv", index=False)

    print(f"Saved: {out_dir / 'model.pkl'}")
    print(f"Saved: {out_dir / 'config.json'}")
    print(f"Saved: {out_dir / 'test_table.pkl.xz'}")
    print(f"Saved: {out_dir / 'predictions.csv'}")
    print(f"Saved: {out_dir / 'qc_report.csv'}")
    print(f"Saved: {out_dir / 'run_info.json'}")
    print(f"Mean sample accuracy: {float((y_test == y_pred).mean()):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TabPFN model from a folder-derived manifest and write HTC-style outputs")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--sheet", default="0_derivative", type=str)
    parser.add_argument("--normalize", action="store_true", help="L1-normalize extracted workbook vectors")
    parser.add_argument("--test-size", default=0.3, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--no-group-split", action="store_true", help="Disable subject-level split and use stratified sample split")
    main(parser.parse_args())
