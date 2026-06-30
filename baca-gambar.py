import cv2

gambar = cv2.imread("kucing.jpg")
cv2.imshow("Gambar Kucing", gambar)

cv2.waitKey(0)
cv2.destroyAllWindows()