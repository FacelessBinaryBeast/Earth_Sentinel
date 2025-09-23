import os

# Paths
image_folder = "D:/aiml project/hackthon/dataset/images/train"
label_folder = "D:/aiml project/hackthon/dataset/labels/train"

# Example bounding boxes (Replace with real data)
bounding_boxes = {
    "img1.jpg": [(0, 0.45, 0.55, 0.30, 0.40)],  # (class_id, x_center, y_center, width, height)
    "img2.jpg": [(0, 0.60, 0.70, 0.20, 0.30)],
}

# Create label files
for image_name, boxes in bounding_boxes.items():
    label_path = os.path.join(label_folder, image_name.replace(".jpg", ".txt"))
    with open(label_path, "w") as f:
        for box in boxes:
            f.write(" ".join(map(str, box)) + "\n")

print("✅ YOLO label files created successfully!")
