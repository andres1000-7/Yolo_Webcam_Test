import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.saved_model.load('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')


# Load label map data (for the COCO dataset)
def load_label_map(label_map_path):
    with open(label_map_path, 'r') as file:
        labels = {}
        current_id = None
        for line in file:
            if 'id:' in line:
                current_id = int(line.split()[1])
            elif 'name:' in line and current_id is not None:
                name = line.split()[1].replace('"', '')
                labels[current_id] = name
                current_id = None
        return labels


labels = load_label_map('mscoco_label_map.pbtxt')


# Function to run inference on an image
def run_inference_for_single_image(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    return detections


# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run object detection
    detections = run_inference_for_single_image(model, image_rgb)

    # Extract detection results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detection_boxes = detections['detection_boxes']
    detection_classes = detections['detection_classes'].astype(np.int64)
    detection_scores = detections['detection_scores']

    # Draw bounding boxes on the frame
    for i in range(num_detections):
        if detection_scores[i] < 0.5:
            continue
        box = detection_boxes[i]
        class_id = detection_classes[i]
        class_name = labels.get(class_id, 'Unknown')

        y_min, x_min, y_max, x_max = box
        y_min = int(y_min * frame.shape[0])
        x_min = int(x_min * frame.shape[1])
        y_max = int(y_max * frame.shape[0])
        x_max = int(x_max * frame.shape[1])

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
