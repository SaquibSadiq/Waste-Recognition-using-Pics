#!/usr/bin/env python3
"""
Simple evaluation script for detect-waste project
"""
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

class SimpleWasteClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Use a simple CNN
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def predict_waste(image_path, model_path, output_path=None):
    """Predict waste in an image"""
    device = torch.device('cpu')
    
    # Load model
    model = SimpleWasteClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Prediction results
    classes = ['No Waste', 'Waste Detected']
    colors = ['green', 'red']
    
    ax2.bar(classes, probabilities[0].numpy(), color=colors)
    ax2.set_title(f'Prediction: {classes[predicted_class]} (Confidence: {confidence:.2f})')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    
    # Add confidence text
    ax2.text(0.5, 0.8, f'Confidence: {confidence:.2f}', 
             transform=ax2.transAxes, ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Prediction saved to: {output_path}")
    else:
        plt.show()
    
    print(f"Prediction: {classes[predicted_class]}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Probabilities: {dict(zip(classes, probabilities[0].numpy()))}")
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Test on sample images
    model_path = "sample_data/output/simple_waste_model.pth"
    
    if not os.path.exists(model_path):
        print("Model not found. Please run training first.")
        exit(1)
    
    # Test on first few sample images
    sample_images = [
        "sample_data/images/sample_000.jpg",
        "sample_data/images/sample_001.jpg", 
        "sample_data/images/sample_002.jpg"
    ]
    
    for i, img_path in enumerate(sample_images):
        if os.path.exists(img_path):
            print(f"\n--- Testing {img_path} ---")
            output_path = f"sample_data/output/prediction_{i}.png"
            predict_waste(img_path, model_path, output_path)
        else:
            print(f"Image not found: {img_path}")
    
    print("\nEvaluation completed!")
