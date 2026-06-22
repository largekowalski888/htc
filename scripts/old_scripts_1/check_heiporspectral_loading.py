from htc import DataPath, settings

data_dir = settings.data_dirs["HeiPorSPECTRAL"]

print("Dataset path HTC is using:")
print(data_dir)

paths = list(DataPath.iterate(data_dir))

print(f"Found {len(paths)} image paths")

if len(paths) == 0:
    raise RuntimeError("No image paths found. Check PATH_Tivita_HeiPorSPECTRAL or dataset structure.")

path = paths[0]

print("First image:")
print(path.image_name())
print("Subject:", path.subject_name)

cube = path.read_cube()
print("Cube shape:", cube.shape)

mask = path.read_segmentation("polygon#annotator1")
print("Mask shape:", mask.shape)
print("Mask labels:", sorted(set(mask.flatten()))[:20])

"""
Run it like this in cmd:
    python scripts/check_heiporspectral_loading.py

Run should produce:
    Dataset path HTC is using:
    /your/path/to/HeiPorSPECTRAL

    Found 5756 image paths
    First image:
    P086#2021_04_15_09_22_02
    Subject: P086
    Cube shape: (480, 640, 100)
    Mask shape: (480, 640)
"""