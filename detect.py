from picamera2 import PiCamera
import cv2
import numpy as np
import RPi.GPIO as GPIO
from time import sleep
from ultralytics import YOLO

# Initialize the YOLOv8 model
model = YOLO('squirrel.pt')

# Initialize GPIO for circuit control
pin_water_gun = 17
pin_turn_right = 18
pin_turn_left = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(pin_water_gun, GPIO.OUT)
GPIO.setup(pin_turn_right, GPIO.OUT)
GPIO.setup(pin_turn_left, GPIO.OUT)

# Initialize Pi Camera
camera = PiCamera()

# Flag for squirrel detection
squirrel_detected = False
photo_count = 0

# Main Loop
while True:
    # Capture frame
    frame = np.empty((480, 640, 3), dtype=np.uint8)
    camera.capture(frame, format="rgb")
    
    # Perform inference
    results = model(frame)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 0:
            squirrel_detected = True
            photo_count += 1
            
            # Start squirting
            GPIO.output(pin_water_gun, GPIO.HIGH)
            sleep(2)
            GPIO.output(pin_water_gun, GPIO.LOW)
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            # Save the image with bounding box
            cv2.imwrite(f'squirrel_detected_{photo_count}.jpg', frame)
            
            print("Squirrel detected")
        else:
            print("Squirrel not detected")
    
    sleep(0.1)  # Pause before capturing the next frame

# Cleanup
GPIO.cleanup()
