# Folder-based Local Modeling Pipeline

This bundle builds a local ML pipeline from class folders similar to `Cat_0007_kidney_deox`, `Cat_0007_kidney_ischem`, etc.

## Assumptions
- Each top-level class root folder name becomes the class label.
- Under each class root, the script looks for `_hypergui_1` folders.
- Each `_hypergui_1` should contain `spectrum_fromCSV*_masked_data.xlsx`.
- Optional `mask.csv` is cataloged but not required for training.

## Typical workflow

### 1) Build a manifest
```powershell
python build_manifest.py --class-roots "C:\data\Cat_0007_kidney_deox;C:\data\Cat_0007_kidney_ischem;C:\data\Cat_0007_kidney_stas" --output-csv "C:\DKFZ\outputs\kidney_manifest.csv"
```

### 2A) Train/test with Random Forest
```powershell
python train_from_manifest_rf.py --manifest "C:\DKFZ\outputs\kidney_manifest.csv" --output-dir "C:\DKFZ\outputs\kidney_rf_run" --sheet 0_derivative --normalize
```

### 2B) Train/test with 1D CNN
```powershell
python train_from_manifest_cnn1d.py --manifest "C:\DKFZ\outputs\kidney_manifest.csv" --output-dir "C:\DKFZ\outputs\kidney_cnn_run" --sheet 0_derivative --normalize
```

### 3) Plot confusion matrix
```powershell
python plot_confusion_matrix.py --run-dir "C:\DKFZ\outputs\kidney_rf_run"
```

## Output artifacts
Both training scripts write:
- `model.pkl` (RF) or `model.pt` (CNN1D)
- `config.json`
- `test_table.pkl.xz`
- `predictions.csv`
- `qc_report.csv`
- `run_info.json`
- `train_manifest.csv`, `test_manifest.csv`

## Notes
- The split is 70/30 by default (`--test-size 0.3`).
- If possible, split is done by `subject_name` to reduce leakage.
- `plot_confusion_matrix.py` uses the existing HTC metric aggregation path.
