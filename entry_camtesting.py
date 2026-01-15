from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")  # use s for testing accuracy

cam1_url = "http://100.114.250.30:8080/video"

cap = cv2.VideoCapture(cam1_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    frame = cv2.resize(frame, (640, 360))

    results = model(frame, conf=0.15)

    annotated = results[0].plot()

    cv2.imshow("ENTRY YOLO TEST", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
