from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from htc import Config, DataPath, DatasetImage, LabelMapping, settings
from htc_projects.atlas.settings_atlas import settings_atlas


def get_features_and_labels(sample, mapping):
    """
    Converts one HTC DatasetImage sample into:
    X: [n_pixels, n_channels]
    y: [n_pixels]
    """

    features = sample["features"]
    labels = sample["labels"]

    # Convert torch tensors to numpy if needed
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    # HTC datasets may store features as C x H x W or H x W x C.
    # We handle both cases.
    if features.ndim == 3:
        if features.shape[0] <= 200:
            # C x H x W -> H x W x C
            features = np.moveaxis(features, 0, -1)

    # Flatten image into pixels
    X = features.reshape(-1, features.shape[-1])
    y = labels.reshape(-1)

    # Keep only valid organ labels
    valid = mapping.is_index_valid(y)

    X = X[valid]
    y = y[valid]

    return X, y


def sample_pixels(X, y, max_pixels_per_image=5000, random_state=42):
    """
    Avoids using every pixel, which can be huge.
    """
    rng = np.random.default_rng(random_state)

    if len(y) <= max_pixels_per_image:
        return X, y

    idx = rng.choice(len(y), size=max_pixels_per_image, replace=False)

    return X[idx], y[idx]


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Basic HTC config for loading already-processed L1 HSI data
    config = Config({
        "input/n_channels": args.n_channels,
        "input/preprocessing": args.preprocessing,
        "input/annotation_name": args.annotation_name,
        "label_mapping": settings_atlas.label_mapping,
    })

    mapping = LabelMapping.from_config(config)
    n_classes = len(mapping)

    data_dir = settings.data_dirs[args.dataset_name]

    # Load all image paths from the dataset folder
    paths = list(DataPath.iterate(data_dir))

    print(f"Found {len(paths)} image paths")

    # Group by subject
    subjects = sorted(set(p.subject_name for p in paths))
    print(f"Subjects: {subjects}")

    # Simple subject split.
    # You can replace this with a proper DataSpecification later.
    train_subjects = set(subjects[:args.n_train_subjects])
    val_subjects = set(subjects[args.n_train_subjects:args.n_train_subjects + args.n_val_subjects])
    test_subjects = set(subjects[args.n_train_subjects + args.n_val_subjects:])

    print(f"Train subjects: {sorted(train_subjects)}")
    print(f"Val subjects:   {sorted(val_subjects)}")
    print(f"Test subjects:  {sorted(test_subjects)}")

    train_paths = [p for p in paths if p.subject_name in train_subjects]
    test_paths = [p for p in paths if p.subject_name in test_subjects]

    train_dataset = DatasetImage(train_paths, train=True, config=config)
    test_dataset = DatasetImage(test_paths, train=False, config=config)

    X_train_all = []
    y_train_all = []

    print("Loading training data...")

    for i, sample in enumerate(train_dataset):
        X, y = get_features_and_labels(sample, mapping)
        X, y = sample_pixels(X, y, args.max_pixels_per_image, args.random_state + i)

        X_train_all.append(X)
        y_train_all.append(y)

        print(f"Train image {i + 1}/{len(train_dataset)}: {X.shape}")

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)

    print(f"Final training matrix: {X_train.shape}")
    print(f"Final training labels: {y_train.shape}")

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    )

    print("Training model...")
    model.fit(X_train, y_train)

    rows = []

    print("Testing model and building subject confusion matrices...")

    for subject in sorted(test_subjects):
        subject_paths = [p for p in test_paths if p.subject_name == subject]
        subject_dataset = DatasetImage(subject_paths, train=False, config=config)

        cms = []
        accuracies = []

        for sample in subject_dataset:
            X_test, y_test = get_features_and_labels(sample, mapping)
            X_test, y_test = sample_pixels(X_test, y_test, args.max_pixels_per_test_image, args.random_state)

            y_pred = model.predict(X_test)

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=list(range(n_classes))
            )

            cms.append(cm)
            accuracies.append(accuracy_score(y_test, y_pred))

        subject_cm = np.sum(np.stack(cms), axis=0)

        rows.append({
            "subject_name": subject,
            "confusion_matrix": subject_cm,
            "accuracy": float(np.mean(accuracies)),
        })

        print(f"{subject}: accuracy={np.mean(accuracies):.3f}")

    df_test = pd.DataFrame(rows)

    output_path = output_dir / "test_table.pkl.xz"
    df_test.to_pickle(output_path)

    # Save config beside it, because your confusion matrix script expects config.json
    config.save_config(output_dir / "config.json")

    print(f"Saved: {output_path}")
    print(f"Saved: {output_dir / 'config.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-name", default="HeiPorSPECTRAL")
    parser.add_argument("--output-dir", default="outputs/baseline_run")

    parser.add_argument("--n-channels", type=int, default=100)
    parser.add_argument("--preprocessing", default="L1")
    parser.add_argument("--annotation-name", default="polygon#annotator1")

    parser.add_argument("--n-train-subjects", type=int, default=7)
    parser.add_argument("--n-val-subjects", type=int, default=1)

    parser.add_argument("--max-pixels-per-image", type=int, default=5000)
    parser.add_argument("--max-pixels-per-test-image", type=int, default=10000)

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    main(args)
    
"""
Run it like this in cmd:
    python scripts/make_test_table_baseline.py \
    --dataset-name HeiPorSPECTRAL \
    --output-dir outputs/baseline_run

Run should produce:
    outputs/baseline_run/config.json
    outputs/baseline_run/test_table.pkl.xz
"""