import os

SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0


class_ids = [
    "without_mask",
    "with_mask",
    "mask_weared_incorrect"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))


# Path to images and annotations
path_images = "./data/images"
path_annot = "./data/annotations"


# Get all XML file paths in path_annot and sort them
xml_files = sorted(
    [
        os.path.join(path_annot, file_name)
        for file_name in os.listdir(path_annot)
        if file_name.endswith(".xml")
    ]
)