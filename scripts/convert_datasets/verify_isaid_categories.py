#!/usr/bin/env python3
"""
Verify that iSAID train and test annotations have consistent category mappings.
"""

import json
import sys
from pathlib import Path

def verify_categories(dataset_dir):
    """Check that train and test have matching category mappings."""
    
    annotations_dir = Path(dataset_dir) / 'annotations'
    train_json = annotations_dir / 'train.json'
    test_json = annotations_dir / 'test.json'
    
    if not train_json.exists():
        print(f"ERROR: {train_json} not found")
        return False
    
    if not test_json.exists():
        print(f"ERROR: {test_json} not found")
        return False
    
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    print("=== TRAIN categories ===")
    for cat in sorted(train_data['categories'], key=lambda x: x['id']):
        print(f"  id={cat['id']}: {cat['name']}")
    
    print("\n=== TEST categories ===")
    for cat in sorted(test_data['categories'], key=lambda x: x['id']):
        print(f"  id={cat['id']}: {cat['name']}")
    
    if train_data['categories'] == test_data['categories']:
        print("\n✓ SUCCESS: Train and test categories match!")
        
        # Sample some annotations to verify
        print("\nSample train annotations:")
        for ann in train_data['annotations'][:5]:
            cat_id = ann['category_id']
            cat_name = next((c['name'] for c in train_data['categories'] if c['id'] == cat_id), 'UNKNOWN')
            print(f"  annotation {ann['id']}: category_id={cat_id} ({cat_name})")
        
        print("\nSample test annotations:")
        for ann in test_data['annotations'][:5]:
            cat_id = ann['category_id']
            cat_name = next((c['name'] for c in test_data['categories'] if c['id'] == cat_id), 'UNKNOWN')
            print(f"  annotation {ann['id']}: category_id={cat_id} ({cat_name})")
        
        return True
    else:
        print("\n✗ ERROR: Train and test categories DO NOT match!")
        print("This will cause incorrect labels in visualizations and evaluation.")
        return False


if __name__ == '__main__':
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else '/localdisk/data3/miguel/datasets/iSAID'
    success = verify_categories(dataset_dir)
    sys.exit(0 if success else 1)
