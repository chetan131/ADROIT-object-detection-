import cv2 
from ultralytics import YOLO 

model = YOLO('yolov8n.pt')  


image_path = 'images.png'  
image = cv2.imread(image_path)


results = model(image)


for result in results:
    boxes = result.boxes  ]
    for box in boxes:

        x1, y1, x2, y2 = box.xyxy[0] 
        confidence = box.conf[0]  #
        class_id = int(box.cls[0])  
        

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'Class {class_id} {confidence:.2f}', (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


output_path = 'output.jpeg'
cv2.imwrite(output_path, image)


cv2.imshow('Detected Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
