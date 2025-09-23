from ultralytics import YOLO

# Load YOLOv8 model with a larger architecture
model = YOLO(r"D:\aiml project\hackthon\yolo-weights\yolov8l.pt")

# Train the model
model.train(
    data=r"D:\aiml project\hackthon\dataset\dataset.yaml",
    imgsz=320, 
    batch=4, 
    epochs=15, 
    workers=0
)


