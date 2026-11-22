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

    # getting the bbox of each result
    for result in  results:
        boxes = result.boxes
        for box in boxes:
            
            """
            #1 bbox with cv2
            x1, y1, x2, y2=box.xyxy[0]
            #-->convert x1, y1, x2, y2 values to integer 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #-->Draw the bouding box
            cv.rectangle(frame,(x1, y1), (x2, y2), (255,0,255),3)
            """
            #2 bbox with cv zone
            x1, y1, x2, y2=box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h= x2-x1, y2-y1
            cvzone.cornerRect(frame, (x1, y1, w, h))

    cv.imshow("yolo", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

