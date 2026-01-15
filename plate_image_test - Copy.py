import cv2
import easyocr

# Load image
img = cv2.imread("car.jpg")

if img is None:
    print("Image not found!")
    exit()

# Show image
cv2.imshow("Car", img)
cv2.waitKey(1000)

# OCR reader
reader = easyocr.Reader(['en'])

# Read text
results = reader.readtext(img)

print("Detected text:")

for (bbox, text, prob) in results:
    print(text, " confidence:", prob)

    # Draw box
    (tl, tr, br, bl) = bbox
    tl = tuple(map(int, tl))
    br = tuple(map(int, br))

    cv2.rectangle(img, tl, br, (0,255,0), 2)
    cv2.putText(img, text, (tl[0], tl[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
