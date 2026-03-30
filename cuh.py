from ultralytics import YOLO
import cv2
import ctypes

# get screen size (once, outside loop)
user32 = ctypes.windll.user32
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)


model = YOLO("yolov10l.pt") #choose model n/s/m/b/l/x

cap = cv2.VideoCapture('samples/test_vid2.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    if w > screen_w or h > screen_h:
        scale = min(screen_w / w, screen_h / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale))) # for resize to fit the display to see

    results = model.track(
        frame,
        persist=True,
        classes=[0, 1, 2, 3, 5, 7],
        conf=0.4,
        iou=0.5,
        device=0,
        max_det=1000
    )

    boxes = results[0].boxes
    names = model.names

    if boxes is not None and boxes.id is not None:
        for b, track_id in zip(boxes, boxes.id):
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cls_id = int(b.cls[0])

            class_name = names[cls_id]
            label = f"{int(track_id)} {class_name}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()