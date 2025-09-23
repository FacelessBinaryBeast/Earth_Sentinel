import os

# Define paths for train and val sets
image_dirs = {
    "train": "C:/Users/ranes/Desktop/hackthon/dataset/images/train",
    "val": "C:/Users/ranes/Desktop/hackthon/dataset/images/val"
}

label_dirs = {
    "train": "C:/Users/ranes/Desktop/hackthon/dataset/labels/train",
    "val": "C:/Users/ranes/Desktop/hackthon/dataset/labels/val"
}

# Ensure label directories exist
for label_dir in label_dirs.values():
    os.makedirs(label_dir, exist_ok=True)

# Define class label (0 since YOLO format uses zero-based indexing)
class_label = 0  

# Generate label files with class name
for dataset_type, image_dir in image_dirs.items():
    label_dir = label_dirs[dataset_type]
    
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):  # Check image formats
            txt_file = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

            # Write label format: class_id x_center y_center width height (placeholders for now)
            with open(txt_file, 'w') as f:
                f.write(f"{class_label} 0.5 0.5 0.4 0.3\n")  # Example YOLO format label

print("✅ Label .txt files created with class 'deforestation' for all images.")
