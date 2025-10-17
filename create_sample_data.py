#!/usr/bin/env python3
"""
Create synthetic sample data for detect-waste project
"""
import json
import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_sample_images(num_images=20, output_dir="sample_data/images"):
    """Create synthetic waste images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample images with random waste objects
    for i in range(num_images):
        # Create random image
        img = Image.new('RGB', (640, 480), color=(100, 150, 100))  # Green background
        draw = ImageDraw.Draw(img)
        
        # Add random "waste" objects (colored rectangles)
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(50, 500)
            y1 = random.randint(50, 400)
            x2 = x1 + random.randint(30, 100)
            y2 = y1 + random.randint(30, 80)
            color = random.choice([(255, 0, 0), (0, 0, 255), (255, 255, 0), (128, 128, 128)])
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        img.save(os.path.join(output_dir, f"sample_{i:03d}.jpg"))

def create_taco_annotations(num_images=20, output_file="sample_data/taco/annotations.json"):
    """Create TACO-style COCO annotations"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # TACO categories (60 categories mapped to detect-waste 7)
    categories = [
        {"id": 1, "name": "Bottle", "supercategory": "litter"},
        {"id": 2, "name": "Cigarette", "supercategory": "litter"},
        {"id": 3, "name": "Paper", "supercategory": "litter"},
        {"id": 4, "name": "Plastic bag", "supercategory": "litter"},
        {"id": 5, "name": "Metal", "supercategory": "litter"},
        {"id": 6, "name": "Glass", "supercategory": "litter"},
        {"id": 7, "name": "Cardboard", "supercategory": "litter"}
    ]
    
    images = []
    annotations = []
    ann_id = 1
    
    for i in range(num_images):
        # Add image info
        img_info = {
            "id": i + 1,
            "file_name": f"sample_{i:03d}.jpg",
            "width": 640,
            "height": 480
        }
        images.append(img_info)
        
        # Add random annotations
        num_objects = random.randint(1, 3)
        for j in range(num_objects):
            x = random.randint(50, 500)
            y = random.randint(50, 400)
            w = random.randint(30, 100)
            h = random.randint(30, 80)
            
            ann = {
                "id": ann_id,
                "image_id": i + 1,
                "category_id": random.randint(1, 7),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            }
            annotations.append(ann)
            ann_id += 1
    
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

def create_epi_annotations(num_images=10, output_file="sample_data/epi/annotations_epi.json"):
    """Create EPI-style annotations"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    categories = [
        {"id": 1, "name": "litter", "supercategory": "litter"}
    ]
    
    images = []
    annotations = []
    ann_id = 1
    
    for i in range(num_images):
        img_info = {
            "id": i + 1,
            "file_name": f"sample_{i:03d}.jpg",
            "width": 640,
            "height": 480
        }
        images.append(img_info)
        
        # Add random litter annotations
        num_objects = random.randint(1, 2)
        for j in range(num_objects):
            x = random.randint(50, 500)
            y = random.randint(50, 400)
            w = random.randint(30, 100)
            h = random.randint(30, 80)
            
            ann = {
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            }
            annotations.append(ann)
            ann_id += 1
    
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

if __name__ == "__main__":
    print("Creating sample images...")
    create_sample_images(20, "sample_data/images")
    
    print("Creating TACO annotations...")
    create_taco_annotations(20, "sample_data/taco/annotations.json")
    
    print("Creating EPI annotations...")
    create_epi_annotations(10, "sample_data/epi/annotations_epi.json")
    
    print("Sample data created successfully!")
    print("Images: sample_data/images/")
    print("TACO annotations: sample_data/taco/annotations.json")
    print("EPI annotations: sample_data/epi/annotations_epi.json")
