from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')  # Replace with your model if different

# Export the model to ONNX format
model.export(format='onnx', opset=12, dynamic=True)  # Opset 12 or higher for compatibility