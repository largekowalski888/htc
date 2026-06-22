import pandas as pd

for name in ["train", "val", "test"]:
    df = pd.read_csv(fr"C:\DKFZ\outputs\kidney_rf_run_valsplit\{name}_manifest.csv")
    print(f"\n{name.upper()} samples per class")
    print(df["class_name"].value_counts())

    print(f"\n{name.upper()} subjects per class")
    print(df.groupby("class_name")["subject_name"].nunique())
