from ultralytics import YOLO
import cv2 as cv
import cvzone # to display all detection
import math
from utils import draw_bbox_conf_cls

video_path = "videos/ppe-1.mp4"
# 1. Open vieo
cap = cv.VideoCapture(video_path) 

if not cap.isOpened():
    print("Error: Could open the video. Check if connected or used by another app.")
    exit()

# 1. Load weights from weight folder
model =YOLO("/yolo-weights/yolov8n.pt") 

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to grab frame.")
        break

    # Optional: Run YOLO detection on each frame
    results = model(frame, stream = True)

    # getting the bbox of each result
    for result in  results:
        boxes = result.boxes
        for box in boxes:
            draw_bbox_conf_cls(box, frame, model)
            
    cv.imshow("yolo", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

