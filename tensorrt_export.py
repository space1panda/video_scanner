import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO


# model = YOLO('yolo8n_clothing.pt')
# model.export(format='onnx')

# Load the ONNX model
onnx_model = onnx.load("models/yolo8n_clothing.onnx")
onnx.checker.check_model(onnx_model)

# # Run inference with ONNX Runtime
ort_session = ort.InferenceSession("models/yolo8n_clothing.onnx")

# # Test with dummy data
dummy_input = torch.rand(1, 3, 640, 640)

outputs = ort_session.run(None, {"images": dummy_input.cpu().numpy()})
print(outputs)
