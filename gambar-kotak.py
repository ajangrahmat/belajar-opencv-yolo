import cv2

gambar = cv2.imread("kucing.jpg")

cv2.rectangle(
    gambar,
    (300, 300), #posisi
    (50,50), #ukuran
    (0,255,0), #warna
    3 #ketebalan
)

cv2.imshow("gambar kotak", gambar)
cv2.waitKey(0)
cv2.destroyAllWindows()