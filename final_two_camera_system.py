from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt") 

cam1_url = "http://100.114.250.30:8080/video"
cam2_url = "http://10.225.198.217:8080/video"
cap1 = cv2.VideoCapture(cam1_url)
cap2 = cv2.VideoCapture(cam2_url)

frame_count = 0

def process(frame):
    global frame_count
    frame_count += 1

    frame = cv2.resize(frame, (640, 360))

    
    if frame_count % 2 != 0:
        return frame

    results = model(frame, conf=0.25, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "car":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                h = y2 - y1
                w = x2 - x1

                py1 = y1 + int(h * 0.45)
                py2 = y1 + int(h * 0.65)
                px1 = x1 + int(w * 0.2)
                px2 = x2 - int(w * 0.2)

                cv2.rectangle(frame, (px1, py1), (px2, py2), (255,0,0), 2)

    return frame

while True:
    r1, f1 = cap1.read()
    r2, f2 = cap2.read()

    if not r1 or not r2:
        break

    f1 = process(f1)
    f2 = process(f2)

    cv2.imshow("ENTRY", f1)
    cv2.imshow("EXIT", f2)

    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
