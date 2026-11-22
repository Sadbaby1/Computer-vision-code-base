from ultralytics import YOLO
import cv2 as cv
import cvzone # to display all detection



# 2. Open camera
cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)



if not cap.isOpened():
    print("Error: Could not open webcam. Check if connected or used by another app.")
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

    cv.imshow("yolo",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

