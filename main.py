import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Function to process the image and perform inference
def process_image(image):
    # Load the YOLO model with the saved weights
    weights_path = 'last.pt'
    model = YOLO(weights_path)

    # Convert the uploaded image to a format that YOLO can process (BGR format for OpenCV)
    img = np.array(image.convert('RGB'))  # Convert to RGB first
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    # Perform inference
    results = model(img)

    # Draw bounding boxes and confidence scores
    for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf):
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red rectangle
        label = f"Confidence: {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img

# Streamlit app layout
st.title('Nepal Vehicle Number Plate Detection')

# Upload image through Streamlit interface
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and process the image
    image = Image.open(uploaded_file)
    processed_img = process_image(image)

    # Convert BGR to RGB for displaying with Streamlit
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    # Display the processed image with bounding boxes and confidence scores
    st.image(processed_img_rgb, caption="Image with Bounding Boxes and Confidence", use_column_width=True)
