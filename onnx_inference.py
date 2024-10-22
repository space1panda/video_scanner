import onnxruntime as ort
import numpy as np
import cv2
import torchvision.transforms as tf
from PIL import Image

# Load the ONNX model
model_path = "yolov8n.onnx"
providers = [
    'CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)
outname = [i.name for i in session.get_outputs()] 
inname = [i.name for i in session.get_inputs()]

image = Image.open('apples.jpg')
print('here')


inp = {inname[0]: tf.ToTensor()(Image.open('apples.jpg')).cpu().numpy()}
outputs = session.run(outname, inp)
print('predicted')

# Preprocess the input image
# def preprocess_image(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     # Resize to the input size of the model (e.g., 640x640)
#     image = cv2.resize(image, (640, 640))
#     # Normalize the image
#     image = image.astype(np.float32) / 255.0
#     # Transpose the image to match the model's input shape (C, H, W)
#     image = np.transpose(image, (2, 0, 1))
#     # Add batch dimension
#     image = np.expand_dims(image, axis=0)
#     return image

# # Run inference
# def run_inference(image):
#     # Define the input name (you might need to check the model's input names)
#     input_name = session.get_inputs()[0].name
#     # Run the model
#     outputs = session.run(None, {input_name: image})
#     return outputs

# # Example usage
# image_path = "apples.jpg"
# input_image = preprocess_image(image_path)
# predictions = run_inference(input_image)

# # Process the predictions (this will depend on your model's output format)

# # Assuming predictions is the output from the ONNX model
# predictions = np.array(predictions)[0]

# # Define thresholds
# confidence_threshold = 0.5  # Set your desired threshold for class probabilities

# # Prepare to store the results
# boxes = []
# scores = []
# classes = []

# # Process each detection
# for i in range(predictions.shape[2]):  # Iterate over detections
#     pred = predictions[0, :, i]  # Get predictions for this detection

#     # Extract bounding box and class probabilities
#     bbox = pred[:4]  # The first four values are the bounding box coordinates
#     class_probs = pred[4:]  # The next 80 values are class probabilities

#     # Get the class with the highest probability
#     max_class_prob = np.max(class_probs)
#     class_id = np.argmax(class_probs)

#     # Filter by confidence threshold
#     if max_class_prob > confidence_threshold:
#         # Convert bbox format if needed, e.g., to [x1, y1, x2, y2]
#         x1, y1, width, height = bbox
#         x2, y2 = x1 + width, y1 + height  # Calculate the bottom right corner
#         boxes.append([x1, y1, x2, y2])
#         scores.append(max_class_prob)
#         classes.append(class_id)

# # Convert lists to numpy arrays for further processing or output
# boxes = np.array(boxes)
# scores = np.array(scores)
# classes = np.array(classes)

# # Print or return the results
# for box, score, class_id in zip(boxes, scores, classes):
#     print(f"Box: {box}, Score: {score}, Class ID: {class_id}")

