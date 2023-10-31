from picamera2 import Picamera2
import torch
import cv2
import numpy as np
import RPi.GPIO as GPIO
from time import sleep
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image 

# Initialize YOLO model
model = YOLO('squirrel.pt')

# Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to square shape
    transforms.Pad(0),  # Additional Padding if needed
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])

# Initialize GPIO
GPIO.setwarnings(False)  # Suppress GPIO warnings
pin_water_gun = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin_water_gun, GPIO.OUT)

# Initialize Pi Camera
picam2 = Picamera2()
config = picam2.preview_configuration
config.main.size = (640, 640)
config.main.format = "RGB888"
picam2.configure(config)
picam2.start()

squirrel_detected = False
photo_count = 0

while True:
    # Capture frame
    buffer = picam2.capture_buffer("main")
    frame = np.array(buffer, copy=False, dtype=np.uint8).reshape((640, 640, 3))

    # Convert to PIL for preprocessing
    pil_frame = Image.fromarray(frame)
    

    # Preprocess
    input_tensor = preprocess(pil_frame)
    input_batch = input_tensor.unsqueeze(0)

    # Perform inference
    results = model(input_batch)
    boxes = results.xyxy[0].cpu().numpy()

    if len(boxes) > 0:
        squirrel_detected = True
        photo_count += 1

        # Activate water gun
        GPIO.output(pin_water_gun, GPIO.HIGH)
        sleep(2)
        GPIO.output(pin_water_gun, GPIO.LOW)

        # Draw bounding boxes and save image
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(f'squirrel_detected_{photo_count}.jpg', frame)

        print("Squirrel detected")
    else:
        print("Squirrel not detected")

    sleep(0.1)

# Cleanup
cv2.destroyAllWindows()
GPIO.cleanup()
