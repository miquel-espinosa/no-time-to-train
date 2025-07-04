import json

# Because the minival json file doesn't have segmentation annotations, we will copy the segmentation annotations from the val json file

MINIVAL = "./data/lvis/lvis_v1_minival_just_bbox.json"
SAVE_MINIVAL = "./data/lvis/lvis_v1_minival.json"
VAL = "./data/lvis/lvis_v1_val.json"

with open(MINIVAL, "r") as f:
    minival = json.load(f)

with open(VAL, "r") as f:
    val = json.load(f)

# Copy the segmentation annotations from the val json file to the minival json file
for ann in minival["annotations"]:
    ann["segmentation"] = val["annotations"][ann["id"]]["segmentation"]

with open(SAVE_MINIVAL, "w") as f:
    json.dump(minival, f)

print(f"Saved minival json to {SAVE_MINIVAL}")