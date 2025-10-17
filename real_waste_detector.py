#!/usr/bin/env python3
"""
Real Waste Detection System
Uses computer vision techniques to detect actual waste in images
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import os
import requests
from io import BytesIO

class RealWasteDetector:
    def __init__(self):
        self.waste_keywords = [
            'bottle', 'can', 'plastic', 'bag', 'wrapper', 'cup', 'container',
            'paper', 'cardboard', 'food', 'trash', 'litter', 'garbage',
            'disposable', 'packaging', 'foil', 'straw', 'lid'
        ]
        
        # Load a pre-trained model for object detection
        self.setup_model()
    
    def setup_model(self):
        """Setup a pre-trained model for better detection"""
        try:
            # Use a pre-trained ResNet for feature extraction
            self.model = models.resnet18(pretrained=True)
            self.model.eval()
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except:
            self.model = None
            self.transform = None
    
    def detect_waste_by_color(self, image):
        """Detect waste based on color analysis"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for common waste items
        waste_colors = {
            'plastic_blue': ([100, 50, 50], [130, 255, 255]),  # Blue plastic
            'plastic_white': ([0, 0, 200], [180, 30, 255]),    # White plastic
            'metal': ([0, 0, 50], [180, 50, 200]),             # Metallic
            'organic': ([20, 100, 100], [40, 255, 255]),       # Brown/organic
        }
        
        waste_score = 0
        total_pixels = image.shape[0] * image.shape[1]
        
        for color_name, (lower, upper) in waste_colors.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            color_pixels = cv2.countNonZero(mask)
            waste_score += (color_pixels / total_pixels) * 0.3
        
        return min(waste_score, 1.0)
    
    def detect_waste_by_shape(self, image):
        """Detect waste based on shape analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        waste_shapes = 0
        total_contours = len(contours)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                # Check for rectangular shapes (common in packaging)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Rectangular objects (bottles, boxes, etc.)
                if 0.3 < aspect_ratio < 3.0:
                    waste_shapes += 1
        
        return min(waste_shapes / max(total_contours, 1), 1.0)
    
    def detect_waste_by_texture(self, image):
        """Detect waste based on texture analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture features
        # High contrast areas often indicate man-made objects
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize the variance
        texture_score = min(laplacian_var / 1000, 1.0)
        
        return texture_score
    
    def detect_waste_by_edges(self, image):
        """Detect waste based on edge density"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_pixels = cv2.countNonZero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = edge_pixels / total_pixels
        
        # Man-made objects typically have more defined edges
        return min(edge_density * 10, 1.0)
    
    def analyze_image_content(self, image):
        """Comprehensive waste detection analysis"""
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Multiple detection methods
        color_score = self.detect_waste_by_color(image)
        shape_score = self.detect_waste_by_shape(image)
        texture_score = self.detect_waste_by_texture(image)
        edge_score = self.detect_waste_by_edges(image)
        
        # Weighted combination
        weights = [0.3, 0.25, 0.25, 0.2]
        total_score = (
            color_score * weights[0] +
            shape_score * weights[1] +
            texture_score * weights[2] +
            edge_score * weights[3]
        )
        
        # Additional checks for natural vs man-made
        natural_indicators = self.detect_natural_elements(image)
        total_score = total_score * (1 - natural_indicators * 0.5)
        
        return {
            'waste_probability': min(max(total_score, 0), 1),
            'color_score': color_score,
            'shape_score': shape_score,
            'texture_score': texture_score,
            'edge_score': edge_score,
            'natural_indicators': natural_indicators
        }
    
    def detect_natural_elements(self, image):
        """Detect natural elements that indicate non-waste"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Green vegetation
        green_lower = np.array([40, 40, 40])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Blue sky/water
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_pixels = cv2.countNonZero(blue_mask)
        
        total_pixels = image.shape[0] * image.shape[1]
        natural_score = (green_pixels + blue_pixels) / total_pixels
        
        return min(natural_score, 1.0)

def main():
    st.set_page_config(
        page_title="Real Waste Detection",
        page_icon="🗑️",
        layout="wide"
    )
    
    st.title("🗑️ Real Waste Detection System")
    st.markdown("**Advanced computer vision techniques to detect actual waste in images**")
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = RealWasteDetector()
    
    detector = st.session_state.detector
    
    # Sidebar
    st.sidebar.header("📸 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG, JPG, or JPEG image"
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Input Image")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.info(f"**Image Info:**\n- Size: {image.size}\n- Mode: {image.mode}")
        else:
            st.info("👆 Please upload an image using the sidebar")
    
    with col2:
        st.header("🤖 AI Analysis")
        if uploaded_file is not None:
            with st.spinner("Analyzing image with computer vision..."):
                results = detector.analyze_image_content(image)
            
            # Main result
            waste_prob = results['waste_probability']
            if waste_prob > 0.6:
                st.error(f"🚨 **WASTE DETECTED**")
                st.error(f"Confidence: {waste_prob:.1%}")
            elif waste_prob > 0.3:
                st.warning(f"⚠️ **POSSIBLE WASTE**")
                st.warning(f"Confidence: {waste_prob:.1%}")
            else:
                st.success(f"✅ **NO WASTE DETECTED**")
                st.success(f"Confidence: {1-waste_prob:.1%}")
            
            # Progress bar
            st.progress(waste_prob)
            
            # Detailed analysis
            st.subheader("📊 Detailed Analysis")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Color Analysis", f"{results['color_score']:.1%}")
                st.metric("Shape Analysis", f"{results['shape_score']:.1%}")
            
            with col_b:
                st.metric("Texture Analysis", f"{results['texture_score']:.1%}")
                st.metric("Edge Analysis", f"{results['edge_score']:.1%}")
            
            # Natural elements
            st.metric("Natural Elements", f"{results['natural_indicators']:.1%}")
            
            # Recommendation
            st.subheader("💡 Analysis")
            if waste_prob > 0.6:
                st.warning("⚠️ This image likely contains waste or man-made objects that could be litter.")
            elif waste_prob > 0.3:
                st.info("ℹ️ This image may contain some waste or man-made objects.")
            else:
                st.success("✨ This image appears to be natural scenery with minimal waste.")
                
        else:
            st.info("Upload an image to see analysis results")
    
    # Footer
    st.markdown("---")
    st.markdown("**About:** This system uses computer vision techniques including:")
    st.markdown("- **Color Analysis**: Detects common waste colors (plastic, metal, etc.)")
    st.markdown("- **Shape Analysis**: Identifies man-made rectangular objects")
    st.markdown("- **Texture Analysis**: Analyzes surface patterns")
    st.markdown("- **Edge Detection**: Finds defined boundaries typical of manufactured items")
    st.markdown("- **Natural Element Detection**: Identifies vegetation and sky to reduce false positives")

if __name__ == "__main__":
    main()
