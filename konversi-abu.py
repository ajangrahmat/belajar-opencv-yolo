import cv2 

gambar = cv2.imread("kucing.jpg")

abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gambar abu", abu)

cv2.waitKey(0)
cv2.destroyAllWindows()