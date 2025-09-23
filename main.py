import cv2
import torch
from ultralytics import YOLO

# Load trained model
model_path = r"D:\aiml project\hackthon\runs\detect\train11\weights\best.pt"
model = YOLO(model_path)

# Load images
image_before_path = r"D:\aiml project\hackthon\input_before.jpg"
image_after_path = r"D:\aiml project\hackthon\input_after.jpg"

image_before = cv2.imread(image_before_path)
image_after = cv2.imread(image_after_path)

if image_before is None or image_after is None:
    print("Error: Could not load one or both images.")
    exit()

# Run detection
results_before = model(image_before)
results_after = model(image_after)

# Debug: Print detection results
print("Detections in BEFORE image:", results_before[0].boxes)
print("Detections in AFTER image:", results_after[0].boxes)

def count_detections(results):
    """Count number of detections in the given results."""
    count = 0
    for result in results:
        if result.boxes is not None and result.boxes.shape[0] > 0:
            count += result.boxes.shape[0]  # Get the number of detected objects
    return count

# Get tree counts
deforestation_before = count_detections(results_before)
deforestation_after = count_detections(results_after)

# Compare results and print conclusion
if deforestation_after > deforestation_before:
    print("🌲 Deforestation detected! Tree cover has decreased in the 'after' image.")
elif deforestation_after < deforestation_before:
    print("🌱 Reforestation detected! Tree cover has increased in the 'after' image.")
else:
    print("✅ No significant change in deforestation detected.")

# Debug: Save annotated images to check if bounding boxes are drawn
before_annotated = results_before[0].plot()
after_annotated = results_after[0].plot()

cv2.imwrite(r"D:\aiml project\hackthon\before_annotated.jpg", before_annotated)
cv2.imwrite(r"D:\aiml project\hackthon\after_annotated.jpg", after_annotated)

# Display images with detections
cv2.imshow("Before", before_annotated)
cv2.imshow("After", after_annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
