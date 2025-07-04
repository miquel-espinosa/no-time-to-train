import json


PATHS = ["./data/lvis/original/lvis_v1_train.json",
         "./data/lvis/original/lvis_v1_minival_just_bbox.json",
         "./data/lvis/original/lvis_v1_val.json"]
OUT_PATHS = ["./data/lvis/lvis_v1_train.json",
             "./data/lvis/lvis_v1_minival_just_bbox.json",
             "./data/lvis/lvis_v1_val.json"]

for PATH, OUT_PATH in zip(PATHS, OUT_PATHS):
    with open(PATH, "r") as f:
        data = json.load(f)

    for i, img in enumerate(data["images"]):
        img["file_name"] = img['coco_url'].split('/')[-1]
        
    # Add is_crowd attribute to the json file
    for i, ann in enumerate(data["annotations"]):
        ann["is_crowd"] = 0
        ann["iscrowd"] = 0

    with open(OUT_PATH, "w") as f:
        json.dump(data, f)
        
    print(f"Processed {PATH} to {OUT_PATH}.")
