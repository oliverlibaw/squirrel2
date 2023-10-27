# Import necessary modules
import picamera
import picamera.array
import numpy as np
import sys
sys.path.append('/home/pi4/project1/.venv/lib/python3.11/site-packages')

from ultralytics import YOLO
import RPi.GPIO as GPIO
from time import sleep

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

# Flag for squirrel detection
squirrel_detected = False
photo_count = 0

# Initialize the camera
with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as output:
        


        while True: #Continuous capture loop
    # Capture an image 
# Initialize Pi Camera
with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as output:
        
        while True:  # Continuous capture loop
            # Capture an image using Pi Camera
            camera.capture(output, 'rgb')
            frame = output.array
            frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

            # Perform inference on the captured frame
            results = model(frame)
            
            # Clear the previous image data in the output
            output.truncate(0)

            # Process the results
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                
                # Check if any boxes are detected
                if len(boxes) > 0:
                    squirrel_detected = True
                    photo_count += 1  # Increment the photo count
                    
                    # Start squirting
                    GPIO.output(pin_water_gun, GPIO.HIGH)
                    sleep(2)  # Squirt for 2 seconds
                    GPIO.output(pin_water_gun, GPIO.LOW)  # Stop squirting
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Save the image with bounding box
                        cv2.imwrite(f'squirrel_detected_{photo_count}.jpg', frame)
                    
                    # Stop turning (you may want to adjust the timing)
                    GPIO.output(pin_turn_right, GPIO.LOW)
                    GPIO.output(pin_turn_left, GPIO.LOW)
                    
                    break  # Exit loop after processing the first detected box

            # Output the detection status
            if squirrel_detected:
                print("Squirrel detected")
            else:
                print("Squirrel not detected")

# Cleanup GPIO settings
GPIO.cleanup()
