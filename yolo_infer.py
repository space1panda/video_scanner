import requests
import numpy as np
import cv2
from ultralytics import YOLO

# Triton server address
url = "http://localhost:8000/v2/models/yolov8"

model = YOLO(url, task='detect')

res = model('test.jpg')

print(res)
