import cv2
import torch
from ultralytics import YOLO
import argparse
import os

# Load trained YOLO model
MODEL_PATH = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)

# Function to detect deforestation in an image
def detect_deforestation(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found!")
        return None

    print(f"🔍 Processing: {image_path}")

    # Run YOLO detection
    results = model.predict(source=image_path, save=True, show=False)  # Show=False for script execution

    # Check if any deforestation is detected
    deforestation_detected = any(len(result.boxes) > 0 for result in results)

    return deforestation_detected

# Function to compare before and after images
def compare_before_after(before_images, after_image):
    print("\n🔎 Analyzing Before Images...")
    before_deforestation = any(detect_deforestation(img) for img in before_images)

    print("\n🔎 Analyzing After Image...")
    after_deforestation = detect_deforestation(after_image)

    # Compare results
    if before_deforestation and after_deforestation:
        print("🟡 Deforestation was present before and remains.")
    elif not before_deforestation and after_deforestation:
        print("🔴 Deforestation has occurred! 🌲❌")
    elif before_deforestation and not after_deforestation:
        print("🟢 Area has recovered, no deforestation detected.")
    else:
        print("✅ No deforestation detected in either images.")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare before and after images for deforestation detection")
    parser.add_argument("before_images", nargs="+", help="Paths to one or more 'before' images")
    parser.add_argument("after_image", type=str, help="Path to the 'after' image")

    args = parser.parse_args()
    
    compare_before_after(args.before_images, args.after_image)
