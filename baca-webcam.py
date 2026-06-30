import cv2

sumber = 1 #index webcam
video = cv2.VideoCapture(sumber)

while True:
    ret, frame = video.read()
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == 27: #tombol esc
        break

video.release()
cv2.destroyAllWindows()