from ultralytics import YOLO


model = YOLO("best28.pt")

results = model.predict("dry_test2.png")

# Iterate over the results
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    class_indices = boxes.cls  # Class indices of the detections
    class_names = [result.names[int(cls)] for cls in class_indices]  # Map indices to names
    print(class_names)