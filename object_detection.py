import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv5 model
model = YOLO('yolov5su.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run object detection
    results = model(image_rgb)

    # Extract detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())  # Ensure class_id is an integer
            class_name = model.names[class_id]

            if confidence < 0.5:
                continue

            # Draw bounding boxes and labels
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Check for 'q' key press to exit
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' or 'Esc' key
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
