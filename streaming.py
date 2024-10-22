import torch
import cv2
from ultralytics import YOLO

# Open a connection to the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

desired_width = 1078
desired_height = 768
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

model = YOLO('yolov8n.pt')
print(model)
print(type(model))


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    with torch.no_grad():
        res = model(frame[..., ::-1])
        frame = res[0].plot()

    # Display the resulting frame
    cv2.imshow('Webcam Stream', frame[..., ::-1])

    # Exit the stream on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
