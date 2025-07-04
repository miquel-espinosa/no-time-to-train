import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
from coco_dataloader import CocoImageDataset
import argparse
from sam2.utils.misc import load_video_frames

def get_args_parser():
    parser = argparse.ArgumentParser("Finetune MLP")
    parser.add_argument("--model", type=str, required=True, help="Model size in ['tiny', 'small', 'base_plus', 'large']")
    parser.add_argument("--save_path", type=str, default='./data/features', help="Path to save the image features")
    parser.add_argument("--split", type=str, default='train2017', help="Split to encode")
    
    return parser.parse_args()


def get_model_cfg(model_name):
    if model_name == 'base_plus':
        model_cfg = "sam2_hiera_b+.yaml"
    else:
        model_cfg = f"sam2_hiera_{model_name[0]}.yaml"
    return model_cfg


def collate_fn(batch):
    images = []
    heights = []
    widths = []
    for image, height, width in batch:
        images.append(image)
        heights.append(height)
        widths.append(width)

    images = torch.stack(images)
    return images, heights, widths

def main(args):

    # Load the SAM2 Hiera model
    model_cfg = get_model_cfg(args.model)
    checkpoint = f"./checkpoints/sam2_hiera_{args.model}.pt"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    
    batch_size = 2
    
    # Create save path
    split = args.split
    save_path = args.save_path
    save_path = os.path.join(save_path, split, args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    # Dataloader
    train_dataset = CocoImageDataset(root=f'./data/{split}',
                                json=f'./data/annotations/instances_{split}.json',
                                image_size=predictor.image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)

    print("Length of dataset: ", len(train_dataset))

    # Iterate over the dataset
    for images, heights, widths in train_loader:
        
        images = images.to(predictor.device) # To float, to cuda. They already include the batch dimension
        
        # ENCODE QUERY IMAGE with image encoder
        backbone_out = predictor.forward_image(images) # Backbone output has keys: 'backbone_fpn', 'vision_pos_enc', 'vision_features'
        _, vision_features, vision_pos_embeds, feature_sizes = predictor._prepare_backbone_features(backbone_out)
        
        # Of course, vision features are specific to each image.
        print("Vision features. Length of list: ", len(vision_features))
        for i, feature in enumerate(vision_features):
            print(f"   Feature {i}. Shape: {feature.shape}, Type: {feature.dtype}")
        
        # Feature sizes are the same in every batch element
        print("Feature sizes: ", feature_sizes)
        
        # Pos embeds are the same in every batch element
        print("Vision pos embeds. Length of list: ", len(vision_pos_embeds))
        for i, pos_embed in enumerate(vision_pos_embeds):
            print(f"   Pos embed {i}. Shape: {pos_embed.shape}, Type: {pos_embed.dtype}")
        for i in range(len(vision_pos_embeds)):
            posembed = vision_pos_embeds[i].permute(1, 0, 2)
            assert posembed[0].sum() == posembed[1].sum()
            
        
        # break
        
        
if __name__ == "__main__":
    args = get_args_parser()
    main(args)
