import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Load models
model = YOLO("yolov8n.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Cameras
ENTRY_CAM = 0
EXIT_CAM = 1   # change to 0 if only one cam

cap1 = cv2.VideoCapture("http://100.87.42.6:8080/video")
cap2 = cv2.VideoCapture("http://10.91.33.217:8080/video")

def read_plate(plate_img):
    result = ocr.ocr(plate_img, cls=True)
    text = ""

    if result:
        for line in result:
            for word in line:
                text += word[1][0] + " "

    return text.strip()

def process_frame(frame, cam_name):
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            if name in ["car", "bus", "truck", "motorcycle"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw vehicle box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

                # Estimate plate region
                w = x2 - x1
                h = y2 - y1

                px1 = int(x1 + 0.2 * w)
                px2 = int(x2 - 0.2 * w)
                py1 = int(y1 + 0.6 * h)
                py2 = int(y1 + 0.85 * h)

                # Crop plate
                plate_img = frame[py1:py2, px1:px2]

                plate_text = ""
                if plate_img.size != 0:
                    plate_text = read_plate(plate_img)

                # Draw plate box
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0,255,255), 2)

                if plate_text != "":
                    cv2.putText(frame, plate_text, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, cam_name, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    return frame

print("Starting ANPR system... Press Q to quit")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Camera error")
        break

    frame1 = process_frame(frame1, "ENTRY CAMERA")
    frame2 = process_frame(frame2, "EXIT CAMERA")

    cv2.imshow("Entry Gate", frame1)
    cv2.imshow("Exit Gate", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
