import cv2 # type: ignore
from ultralytics import YOLO # type: ignore

# Load the YOLOv8 model (you can specify 'yolov8n.pt' for the small version)
model = YOLO('yolov8n.pt')  # Replace with your model path if necessary

# Load the image
image_path = 'images.png'  # Update with your local image path
image = cv2.imread(image_path)

# Perform detection
results = model(image)

# Process results
for result in results:
    boxes = result.boxes  # Get the bounding boxes
    for box in boxes:
        # Draw bounding box and label on the image
        x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
        confidence = box.conf[0]  # Confidence score
        class_id = int(box.cls[0])  # Class ID
        
        # Draw the rectangle on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'Class {class_id} {confidence:.2f}', (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Save or display the output image
output_path = 'output.jpeg'
cv2.imwrite(output_path, image)

# Optional: Display the output image
cv2.imshow('Detected Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
