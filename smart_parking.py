import cv2
from ultralytics import YOLO
import easyocr
from datetime import datetime

# ========== SETTINGS ==========
ENTRY_CAM = "http://10.91.33.217:8080/video"
EXIT_CAM  = "http://100.77.51.41:8080/video"

TOTAL_SLOTS = 20

# ==============================

model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=False)

cap_entry = cv2.VideoCapture(ENTRY_CAM)
cap_exit  = cv2.VideoCapture(EXIT_CAM)

parked_cars = {}   # plate -> entry_time

def read_plate(img):
    results = reader.readtext(img)
    if len(results) == 0:
        return None
    results = sorted(results, key=lambda x: x[2], reverse=True)
    text = results[0][1]
    text = text.replace(" ", "").upper()
    return text

def log(msg):
    with open("parking_log.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)

frame_count = 0

while True:
    ret1, frame1 = cap_entry.read()
    ret2, frame2 = cap_exit.read()

    if not ret1 or not ret2:
        print("Camera error")
        break

    frame_count += 1

    # Skip frames to reduce lag
    if frame_count % 5 != 0:
        cv2.imshow("ENTRY", frame1)
        cv2.imshow("EXIT", frame2)
        if cv2.waitKey(1) == 27:
            break
        continue

    # ========== ENTRY CAMERA ==========
    results1 = model(frame1, conf=0.4)
    for r in results1:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            plate_img = frame1[y1:y2, x1:x2]

            plate_text = read_plate(plate_img)

            cv2.rectangle(frame1,(x1,y1),(x2,y2),(0,255,0),2)

            if plate_text:
                cv2.putText(frame1, plate_text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                if plate_text not in parked_cars and len(parked_cars) < TOTAL_SLOTS:
                    t = datetime.now().strftime("%H:%M:%S")
                    parked_cars[plate_text] = t
                    log(f"{plate_text} ENTER {t}")

    # ========== EXIT CAMERA ==========
    results2 = model(frame2, conf=0.4)
    for r in results2:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            plate_img = frame2[y1:y2, x1:x2]

            plate_text = read_plate(plate_img)

            cv2.rectangle(frame2,(x1,y1),(x2,y2),(0,0,255),2)

            if plate_text:
                cv2.putText(frame2, plate_text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

                if plate_text in parked_cars:
                    t = datetime.now().strftime("%H:%M:%S")
                    log(f"{plate_text} EXIT  {t}")
                    del parked_cars[plate_text]

    # ========== DISPLAY INFO ==========
    free = TOTAL_SLOTS - len(parked_cars)

    cv2.putText(frame1, f"FREE SLOTS: {free}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("ENTRY", frame1)
    cv2.imshow("EXIT", frame2)

    if cv2.waitKey(1) == 27:   # ESC to quit
        break

cap_entry.release()
cap_exit.release()
cv2.destroyAllWindows()
