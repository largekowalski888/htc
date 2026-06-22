
from htc import Config, DataPath, DatasetImage, LabelMapping
from htc_projects.atlas.settings_atlas import settings_atlas

config = Config({
    "input/n_channels": 100,
    "input/preprocessing": "L1",
    "input/annotation_name": "polygon#annotator1",
    "label_mapping": settings_atlas.label_mapping,
})

paths = [
    DataPath.from_image_name("P086#2021_04_15_09_22_02"),
    DataPath.from_image_name("P093#2021_04_28_08_49_12"),
]

dataset = DatasetImage(paths, train=False, config=config)
sample = dataset[0]

print(sample.keys())
print("features:", sample["features"].shape)
print("labels:", sample["labels"].shape)
print("valid_pixels:", sample["valid_pixels"].shape)

mapping = LabelMapping.from_config(config)
print("Unique label indices:", sample["labels"].unique())
