#!/usr/bin/env python3
"""
Waste Detection Web App
Upload images and get waste detection results
"""
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import io
import os

class SimpleWasteClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
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

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = "sample_data/output/simple_waste_model.pth"
    if not os.path.exists(model_path):
        st.error("Model not found! Please run training first.")
        return None
    
    model = SimpleWasteClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_waste(image, model):
    """Predict waste in uploaded image"""
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    classes = ['No Waste', 'Waste Detected']
    return classes[predicted_class], confidence, probabilities[0].numpy()

def main():
    st.set_page_config(
        page_title="Waste Detection App",
        page_icon="🗑️",
        layout="wide"
    )
    
    st.title("🗑️ Waste Detection App")
    st.markdown("Upload an image to detect waste using AI!")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG, JPG, or JPEG image"
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Uploaded Image")
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Get image info
            st.info(f"**Image Info:**\n- Size: {image.size}\n- Mode: {image.mode}\n- Format: {uploaded_file.type}")
        else:
            st.info("👆 Please upload an image using the sidebar")
            # Show sample images
            st.subheader("📁 Sample Images")
            sample_dir = "sample_data/images"
            if os.path.exists(sample_dir):
                sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:3]
                for i, sample_file in enumerate(sample_files):
                    sample_path = os.path.join(sample_dir, sample_file)
                    sample_img = Image.open(sample_path)
                    st.image(sample_img, caption=f"Sample {i+1}: {sample_file}", width=200)
    
    with col2:
        st.header("🤖 AI Prediction")
        if uploaded_file is not None:
            # Make prediction
            with st.spinner("Analyzing image..."):
                prediction, confidence, probabilities = predict_waste(image, model)
            
            # Display results
            if prediction == "Waste Detected":
                st.error(f"🚨 **{prediction}**")
                st.error(f"Confidence: {confidence:.1%}")
            else:
                st.success(f"✅ **{prediction}**")
                st.success(f"Confidence: {confidence:.1%}")
            
            # Confidence bar
            st.progress(confidence)
            
            # Detailed probabilities
            st.subheader("📊 Detailed Results")
            classes = ['No Waste', 'Waste Detected']
            colors = ['green', 'red']
            
            for i, (cls, prob, color) in enumerate(zip(classes, probabilities, colors)):
                st.metric(
                    label=cls,
                    value=f"{prob:.1%}",
                    delta=f"{prob-confidence:.1%}" if i != (1 if prediction == "Waste Detected" else 0) else None
                )
            
            # Recommendation
            st.subheader("💡 Recommendation")
            if prediction == "Waste Detected":
                st.warning("⚠️ This image contains waste. Consider proper disposal methods.")
            else:
                st.info("✨ This image appears clean! Keep up the good work.")
                
        else:
            st.info("Upload an image to see AI predictions here")
    
    # Footer
    st.markdown("---")
    st.markdown("**About:** This app uses a trained CNN model to detect waste in images.")
    st.markdown("**Model:** Simple CNN with 101K parameters, trained on synthetic waste data.")
    
    # Add some sample predictions for demo
    if uploaded_file is None:
        st.subheader("🎯 Try These Sample Images")
        sample_dir = "sample_data/images"
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:3]
            
            cols = st.columns(len(sample_files))
            for i, (col, sample_file) in enumerate(zip(cols, sample_files)):
                with col:
                    sample_path = os.path.join(sample_dir, sample_file)
                    sample_img = Image.open(sample_path)
                    
                    # Make prediction for sample
                    pred, conf, probs = predict_waste(sample_img, model)
                    
                    st.image(sample_img, caption=sample_file, width=150)
                    if pred == "Waste Detected":
                        st.error(f"🚨 {pred}\n({conf:.0%})")
                    else:
                        st.success(f"✅ {pred}\n({conf:.0%})")

if __name__ == "__main__":
    main()
