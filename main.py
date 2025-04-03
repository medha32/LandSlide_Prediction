import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# Load YOLO model
model_path = "last.pt"  # Ensure this file is in the working directory
model = YOLO(model_path)

# Streamlit UI
st.title("Landslide Detection using YOLO")
st.sidebar.header("Upload an Image or Video")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose an image or video", type=["jpg", "png", "jpeg", "mp4", "avi", "mov"])

# Zoom slider
zoom = st.sidebar.slider("Zoom Level", 1.0, 3.0, 1.5, 0.1)

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

def process_image(image):
    img = np.array(image)
    results = model(img, conf=confidence_threshold)
    result_img = results[0].plot()  # Get the image with predictions
    return result_img

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    temp_video_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=confidence_threshold)
        frame = results[0].plot()
        out.write(frame)
    
    cap.release()
    out.release()
    return temp_video_path

if uploaded_file:
    file_type = uploaded_file.type.split("/")[0]
    if file_type == "image":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        processed_image = process_image(image)
        st.image(processed_image, caption="Landslide Detection Output", use_column_width=True)
    
    elif file_type == "video":
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())
        temp_video.close()
        
        st.video(temp_video.name)
        st.write("Processing video... This may take some time.")
        output_video_path = process_video(temp_video.name)
        
        st.video(output_video_path)
        os.remove(temp_video.name)  # Clean up temporary file
