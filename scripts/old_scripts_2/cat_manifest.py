from pathlib import Path
import pandas as pd

roots = [
    Path(r"C:\DKFZ\MLData\data\Cat_Pig\Cat_atlas\Cat_0007_kidney"),
    Path(r"C:\DKFZ\MLData\data\Cat_Pig\Cat_atlas_deox\Cat_0007_kidney_deox"),
    Path(r"C:\DKFZ\MLData\data\Cat_Pig\Cat_atlas_deox\Cat_0007_kidney_ischem"),
    Path(r"C:\DKFZ\MLData\data\Cat_Pig\Cat_atlas_deox\Cat_0007_kidney_stas"),
]

rows = []

for root in roots:
    condition = root.name.replace("Cat_0007_kidney", "").strip("_")
    if condition == "":
        condition = "baseline"

    for hypergui_dir in root.rglob("_hypergui_1"):
        sample_dir = hypergui_dir.parent
        mask_path = hypergui_dir / "mask.csv"

        spectrum_files = list(hypergui_dir.glob("spectrum_fromCSV*.xlsx"))
        if not mask_path.exists() or not spectrum_files:
            continue

        subject_name = sample_dir.parent.name     # e.g. P044_Pig003_...
        timestamp = sample_dir.name               # e.g. 2020_02_01_10_23_29

        rows.append({
            "sample_dir": str(sample_dir),
            "subject_name": subject_name,
            "timestamp": timestamp,
            "condition": condition,
            "mask_path": str(mask_path),
            "spectrum_path": str(spectrum_files[0]),
        })

df = pd.DataFrame(rows)
df.to_csv("cat_manifest.csv", index=False)
print(df.head())
print("n_samples =", len(df))
