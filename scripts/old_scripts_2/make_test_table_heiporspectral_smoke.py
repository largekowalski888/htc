from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from htc import Config, DataPath, DatasetImage, LabelMapping
from htc_projects.atlas.settings_atlas import settings_atlas


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    # Output run directory
    run_dir = Path(r"C:\DKFZ\outputs\heiporspectral_smoke_run")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Minimal HTC config for labeled HeiPorSPECTRAL samples
    config = Config({
        "input/n_channels": 100,
        "input/preprocessing": "L1",
        "input/annotation_name": "polygon#annotator1",
        "label_mapping": settings_atlas.label_mapping,
    })

    mapping = LabelMapping.from_config(config)
    n_classes = len(mapping)

    # Use the 2-image HeiPorSPECTRAL example subset
    paths = [
        DataPath.from_image_name("P086#2021_04_15_09_22_02"),
        DataPath.from_image_name("P093#2021_04_28_08_49_12"),
    ]

    dataset = DatasetImage(paths, train=False, config=config)

    rows = []

    for p, sample in zip(paths, dataset, strict=True):
        labels = to_numpy(sample["labels"])
        valid_pixels = to_numpy(sample["valid_pixels"]).astype(bool)

        # Use only annotated pixels
        y_true = labels[valid_pixels].reshape(-1)

        # Perfect smoke-test predictions:
        y_pred = y_true.copy()

        # If you want slight non-perfect behavior later, uncomment:
        rng = np.random.default_rng(42)
        flip_idx = rng.choice(len(y_pred), size=max(1, len(y_pred)//20), replace=False)
        y_pred[flip_idx] = (y_pred[flip_idx] + 1) % n_classes

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(n_classes))
        )

        acc = accuracy_score(y_true, y_pred)

        rows.append({
            "subject_name": p.subject_name,
            "image_name": p.image_name(),
            "confusion_matrix": cm,
            "accuracy": float(acc),
        })

        print(f"{p.image_name()} -> subject={p.subject_name}, valid_pixels={len(y_true)}, accuracy={acc:.3f}")

    df = pd.DataFrame(rows)

    # Save HTC-style run artifacts
    df.to_pickle(run_dir / "test_table.pkl.xz")
    config.save_config(run_dir / "config.json")

    print(f"\nSaved: {run_dir / 'test_table.pkl.xz'}")
    print(f"Saved: {run_dir / 'config.json'}")


if __name__ == "__main__":
    main()
