import os
from PIL import Image

"""
This script is used to resize the images to 512x512 and save them to a new directory.

python3 no_time_to_train/make_plots/paper_fig_square_imgs.py

This is for the paper figures.
"""

train_image_ids = [
    "000000536752",
    "000000119304",
    "000000174276",
    "000000305752",
    "000000538938",
    "000000282711",
    "000000546723",
    "000000513659",
    "000000509037",
    "000000143540",
]

# List of image IDs
val_image_ids = [
    "000000000139",
    "000000001268",
    "000000001353",
    "000000002153",
    "000000002299",
    "000000003255",
    "000000004495",
    "000000005193",
    "000000005992",
    "000000006723",
    "000000007281",
    "000000007511",
    "000000007816",
    "000000008021",
    "000000009483",
    "000000010363",
    "000000012120",
]

# Define paths
output_dir = "./no_time_to_train/make_plots/square_images"
train_output_dir = os.path.join(output_dir, "train")
val_output_dir = os.path.join(output_dir, "val")

# Create output directories if they don't exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Process train images
train_input_dir = "./data/coco/train2017"
for img_id in train_image_ids:
    # Construct input and output paths
    input_path = os.path.join(train_input_dir, f"{img_id}.jpg")
    output_path = os.path.join(train_output_dir, f"{img_id}.jpg")
    
    # Load image
    img = Image.open(input_path)
    
    # Resize to 512x512 (this will distort non-square images)
    img_resized = img.resize((512, 512), Image.BICUBIC)
    
    # Save the resized image
    img_resized.save(output_path)
    print(f"Processed train {img_id}.jpg")

# Process val images
val_input_dir = "./data/coco/val2017"
for img_id in val_image_ids:
    # Construct input and output paths
    input_path = os.path.join(val_input_dir, f"{img_id}.jpg")
    output_path = os.path.join(val_output_dir, f"{img_id}.jpg")
    
    # Load image
    img = Image.open(input_path)
    
    # Resize to 512x512 (this will distort non-square images)
    img_resized = img.resize((512, 512), Image.BICUBIC)
    
    # Save the resized image
    img_resized.save(output_path)
    print(f"Processed val {img_id}.jpg")

print(f"All images saved to {output_dir}")
