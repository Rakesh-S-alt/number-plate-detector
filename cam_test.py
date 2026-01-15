import cv2

url = "http://100.126.221.205:8080/video"  # put your phone IP here

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame")
        break

    cv2.imshow("Phone Camera", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
