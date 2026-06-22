# Task-based Spectral Modeling Framework

This folder contains a minimal task-driven framework for:
- training models from spectral tables
- predicting on CAT manifests
- evaluating into HTC-style `test_table.pkl.xz` artifacts
- plotting confusion matrices

## Folder structure

```text
scripts/
  data_adapters/
  tasks/
  train_task.py
  predict_task.py
  evaluate_task.py
  plot_confusion_matrix.py
```

## Typical usage

1. Train
```powershell
python train_task.py --task tasks\organ_20class.json
```

2. Predict
```powershell
python predict_task.py --run-dir C:\DKFZ\outputs\task_runs\organ_20class
```

3. Evaluate
```powershell
python evaluate_task.py --run-dir C:\DKFZ\outputs\task_runs\organ_20class
```

4. Plot
```powershell
python plot_confusion_matrix.py --run-dir C:\DKFZ\outputs\task_runs\organ_20class
```
