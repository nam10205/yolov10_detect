from ultralytics import YOLO
import cv2

model = YOLO("yolov10l.pt")

cap = cv2.VideoCapture('test_vid.avi')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0, 1, 2, 3, 5, 7],
        conf=0.4,
        iou=0.5,
        device=0
    )

    annotated = results[0].plot()
    cv2.imshow("Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()