# No code yet!


import streamlit as st
import cv2
import requests
import numpy as np
import tempfile
from PIL import Image

# Streamlit page configuration
st.title("Real-Time Object Detection with YOLOv8")

# Start webcam
start_button = st.button('Start Webcam')

# Placeholder for webcam feed
frame_placeholder = st.empty()

# Function to perform inference
def infer_image(image):
    url = "https://api.ultralytics.com/v1/predict/XoFWONj7The3OBQbQt4u"
    headers = {"x-api-key": "21e08ad9cc673c4364305135c69743a8fcdbf6da0e"}
    data = {"size": 640, "confidence": 0.25, "iou": 0.45}
    response = requests.post(url, headers=headers, data=data, files={"image": image})
    response.raise_for_status()
    return response.json()

# Function to overlay bounding boxes
def draw_boxes(img, detections):
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection['box']
        label = detection['label']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img

if start_button:
    cap = cv2.VideoCapture(0)  # Use 0 for web camera

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break
        
        # Convert the frame to PIL Image
        frame_pil = Image.fromarray(frame)
        
        # Convert PIL Image to a bytes object
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            frame_pil.save(tmp_file, format="JPEG")
            tmp_file.seek(0)  # Move the cursor to the beginning of the file
            # Perform inference
            results = infer_image(tmp_file)
            
            # Draw bounding boxes on the original frame
            detections = results.get("predictions", [])
            frame_with_boxes = draw_boxes(frame, detections)
            
            # Display the frame with bounding boxes
            frame_placeholder.image(frame_with_boxes, channels="BGR", use_column_width=True)

# Cleanup
if 'cap' in locals():
    cap.release()
