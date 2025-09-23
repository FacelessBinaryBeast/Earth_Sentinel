import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
from ultralytics import YOLO

# Initialize Tkinter (hide main window)
root = tk.Tk()
root.withdraw()

# Load trained model
model_path = r"D:\aiml project\hackthon\runs\detect\train11\weights\best.pt"
model = YOLO(model_path)

# Open file dialog for selecting images
image_before_path = filedialog.askopenfilename(title="Select the 'Before' image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
image_after_path = filedialog.askopenfilename(title="Select the 'After' image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])

# Check if user selected images
if not image_before_path or not image_after_path:
    print("❌ Error: No image selected.")
    exit()

# Load images
image_before = cv2.imread(image_before_path)
image_after = cv2.imread(image_after_path)

if image_before is None or image_after is None:
    print("❌ Error: Could not load one or both images. Please check the files.")
    exit()

# Convert images to RGB (YOLO expects RGB)
image_before = cv2.cvtColor(image_before, cv2.COLOR_BGR2RGB)
image_after = cv2.cvtColor(image_after, cv2.COLOR_BGR2RGB)

# Run detection with confidence filtering
confidence_threshold = 0.5  # Only count high-confidence detections
results_before = model.predict(image_before, conf=confidence_threshold)
results_after = model.predict(image_after, conf=confidence_threshold)

# Function to count detections above confidence threshold
def count_detections(results):
    count = 0
    for result in results:
        for box in result.boxes:
            if box.conf[0] >= confidence_threshold:  # Count only high-confidence detections
                count += 1
    return count

# Count tree detections
trees_before = count_detections(results_before)
trees_after = count_detections(results_after)

# Check for deforestation
if trees_after < trees_before:
    print("No deforestation detected.")
else:
    print("Deforestation detected! Tree cover has decreased in the 'after' image.")

# Show detection results
cv2.imshow("Before", results_before[0].plot())
cv2.imshow("After", results_after[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()


