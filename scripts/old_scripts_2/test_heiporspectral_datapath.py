from htc import DataPath

path = DataPath.from_image_name("P086#2021_04_15_09_22_02")

print("Path:", path)
print("Image name:", path.image_name())
print("Cube shape:", path.read_cube().shape)

seg = path.read_segmentation("polygon#annotator1")
print("Seg shape:", seg.shape, "dtype:", seg.dtype)

print("Meta label_meta:", path.meta("label_meta"))
