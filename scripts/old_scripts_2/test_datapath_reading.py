
from pathlib import Path
from htc import DataPath

sample_dir = Path(
    r"C:\DKFZ\HeiPorSPECTRAL_example\data\subjects\P086\2021_04_15_09_22_02"
)

print("Exists:", sample_dir.exists())
print("Path:", sample_dir)

p = DataPath(sample_dir)

print("\n--- BASIC DATAPATH INFO ---")
print("DataPath object:", p)
print("Image name:", p.image_name())

print("\n--- META ---")
try:
    meta = p.meta()
    print("Meta keys:", list(meta.keys())[:20] if isinstance(meta, dict) else meta)
except Exception as e:
    print("meta() failed:", e)

print("\n--- SEGMENTATION ---")
try:
    seg = p.read_segmentation(annotation_name="all")
    if isinstance(seg, dict):
        print("Segmentation keys:", list(seg.keys()))
        for k, v in seg.items():
            try:
                print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            except Exception:
                print(f"{k}: {type(v)}")
    else:
        try:
            print("Segmentation shape:", seg.shape, "dtype:", seg.dtype)
        except Exception:
            print("Segmentation:", type(seg))
except Exception as e:
    print("read_segmentation() failed:", e)

print("\n--- CUBE ---")
try:
    cube = p.read_cube()
    print("Cube shape:", cube.shape, "dtype:", cube.dtype)
except Exception as e:
    print("read_cube() failed:", e)