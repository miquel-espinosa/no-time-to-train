import json

PATHS = [
    "./data/pascal_voc/annotations/long_filenames/voc0712_train.json",
    "./data/pascal_voc/annotations/long_filenames/voc0712_trainval.json",
    "./data/pascal_voc/annotations/long_filenames/voc0712_val.json",
    "./data/pascal_voc/annotations/long_filenames/voc07_test.json",
    "./data/pascal_voc/annotations/long_filenames/voc07_train.json",
    "./data/pascal_voc/annotations/long_filenames/voc07_trainval.json",
    "./data/pascal_voc/annotations/long_filenames/voc07_val.json",
    "./data/pascal_voc/annotations/long_filenames/voc12_train.json",
    "./data/pascal_voc/annotations/long_filenames/voc12_trainval.json",
    "./data/pascal_voc/annotations/long_filenames/voc12_val.json",
]

OUT_PATHS = [
    "./data/pascal_voc/annotations/voc0712_train.json",
    "./data/pascal_voc/annotations/voc0712_trainval.json",
    "./data/pascal_voc/annotations/voc0712_val.json",
    "./data/pascal_voc/annotations/voc07_test.json",
    "./data/pascal_voc/annotations/voc07_train.json",
    "./data/pascal_voc/annotations/voc07_trainval.json",
    "./data/pascal_voc/annotations/voc07_val.json",
    "./data/pascal_voc/annotations/voc12_train.json",
    "./data/pascal_voc/annotations/voc12_trainval.json",
    "./data/pascal_voc/annotations/voc12_val.json",
]
for PATH, OUT_PATH in zip(PATHS, OUT_PATHS):
    with open(PATH, "r") as f:
        data = json.load(f)

    for i, img in enumerate(data["images"]):
        img["file_name"] = img['file_name'].split('/')[-1]
        
    with open(OUT_PATH, "w") as f:
        json.dump(data, f)
        
    print(f"Processed {PATH} to {OUT_PATH}.")
