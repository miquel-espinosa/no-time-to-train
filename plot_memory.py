import os
import json
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict


DATASETS = ["coco", "NEU-DET", "DIOR", "FISH", "UODD", "ArTaxOr", "clipart1k"]
SHOTS = [1, 5, 10]
SAVE_DIR = "results_analysis/memory_visualization/"
FONT_PATH = "RobotoMono-MediumItalic.ttf"


def find_free_corner(bbox, img_width, img_height, text_width, text_height, margin=10):
    x, y, width, height = bbox
    corners = [
        (margin, margin),  # top-left
        (img_width - text_width - margin, margin),  # top-right
        (margin, img_height - text_height - margin),  # bottom-left
        (img_width - text_width - margin, img_height - text_height - margin)  # bottom-right
    ]
    
    for corner_x, corner_y in corners:
        # Check if the text box overlaps with the bbox
        if not (corner_x < x + width and corner_x + text_width > x and
                corner_y < y + height and corner_y + text_height > y):
            return corner_x, corner_y
    
    # If all corners overlap, return top-left as default
    return corners[0]


def process_dataset(dataset_name, shot):
    # Load the JSON data
    json_file = os.path.join("data", dataset_name, "annotations", f"{shot}_shot.json")
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"JSON file {json_file} not found. Skipping.")
        return

    # Create output directory
    output_dir = os.path.join(SAVE_DIR, f"{dataset_name}_{shot}shot")
    os.makedirs(output_dir, exist_ok=True)

    # Create mappings
    image_id_to_file = {img["id"]: img["file_name"] for img in json_data["images"]}
    # Create category mapping
    categories = {cat["id"]: cat["name"] for cat in json_data["categories"]}
    
    # Counter for images per category
    category_counters = defaultdict(int)

    # Process each annotation
    for annotation in json_data["annotations"]:
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        category_name = categories[category_id]
        
        # Increment counter for this category
        category_counters[category_name] += 1
        
        # Load the corresponding image
        image_file = image_id_to_file[image_id]
        image_path = os.path.join("data", dataset_name, "train", image_file)
        try:
            with Image.open(image_path) as img:
                # Calculate line width as 1% of image width
                line_width = max(1, int(img.width * 0.01))
                
                font_size = int(min(img.width, img.height) * 0.1)
                try:
                    font = ImageFont.truetype(FONT_PATH, size=font_size)
                except Exception as e:
                    print(f"Error loading font: {e}. Using default font.")
                    font = ImageFont.load_default()
                
                draw = ImageDraw.Draw(img)
                
                # Draw the bounding box
                x, y, width, height = bbox
                draw.rectangle([x, y, x + width, y + height], outline="red", width=line_width)
                
                # Get text dimensions
                text_bbox = draw.textbbox((0, 0), category_name, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Find suitable corner for text
                text_x, text_y = find_free_corner(bbox, img.width, img.height, text_width, text_height)
                
                # Draw text directly without background
                draw.text((text_x, text_y), category_name, fill="red", font=font)
                
                # Save the image with overlay using category name and counter
                file_ext = os.path.splitext(image_file)[1]  # Get original file extension
                output_filename = f"{category_name}_{shot}_shot_{category_counters[category_name]}{file_ext}"
                output_path = os.path.join(output_dir, output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create subdirectories if needed
                img.save(output_path)
        except FileNotFoundError:
            print(f"Image file {image_path} not found. Skipping.")

    print(f"Processed images for {dataset_name} {shot}-shot saved to {output_dir}.")


def main():
    # Create base save directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Process each dataset and shot combination
    for dataset in DATASETS:
        for shot in SHOTS:
            print(f"\nProcessing {dataset} with {shot} shot(s)...")
            process_dataset(dataset, shot)


if __name__ == "__main__":
    main()
