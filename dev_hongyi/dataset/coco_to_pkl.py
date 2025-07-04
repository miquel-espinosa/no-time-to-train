import sys
import json
import pickle
from collections import OrderedDict
import random

# Set random seed for reproducibility
random.seed(42)

def convert_coco_to_pkl(json_path, output_path, target_examples):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create an OrderedDict structure
    converted_data = OrderedDict()
    for ann in data['annotations']:
        cat_id = ann['category_id']
        if cat_id not in converted_data:
            converted_data[cat_id] = []
        
        # Check if image entry already exists for this category
        existing_entry = next((entry for entry in converted_data[cat_id] if entry['img_id'] == ann['image_id']), None)
        if existing_entry:
            existing_entry['ann_ids'].append(ann['id'])
        else:
            converted_data[cat_id].append({'img_id': ann['image_id'], 'ann_ids': [ann['id']]})
    
    # Process each category to duplicate entries with multiple annotations
    for cat_id in converted_data:
        new_entries = []
        for entry in converted_data[cat_id]:
            if len(entry['ann_ids']) > 1:
                # If an entry has multiple annotations, duplicate it
                num_duplicates = len(entry['ann_ids'])
                for _ in range(num_duplicates):
                    new_entries.append(entry.copy())
            else:
                new_entries.append(entry)
        converted_data[cat_id] = new_entries
    
    # Check if the number of examples per category is less than target_examples
    for cat_id in converted_data:
        if len(converted_data[cat_id]) < target_examples:
            print(f"Category {cat_id} has only {len(converted_data[cat_id])} examples. Target is {target_examples}.")
            # Randomly duplicate entries until we reach target_examples
            additional_needed = target_examples - len(converted_data[cat_id])
            additional_entries = random.choices(converted_data[cat_id], k=additional_needed)
            converted_data[cat_id].extend(additional_entries)
    
    # Write to .pkl file
    with open(output_path, 'wb') as f:
        pickle.dump(converted_data, f)
    
    print(f"Converted data written to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_json_path> <output_pkl_path> <target_examples(int)>")
        sys.exit(1)
    
    input_json_path = sys.argv[1]
    output_pkl_path = sys.argv[2]
    target_examples = int(sys.argv[3])
    
    convert_coco_to_pkl(input_json_path, output_pkl_path, target_examples)
