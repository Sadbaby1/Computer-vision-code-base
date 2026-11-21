from ultralytics import YOLO
import cv2 as cv

model = YOLO('/yolo-weights/yolov8n.pt')
results = model("Images/3.png", show=True)
cv.waitKey(0)
annotated_frame = results[0].plot()

cv.imshow("YOLO Detection", annotated_frame)
cv.waitKey(0)
cv.destroyAllWindows()