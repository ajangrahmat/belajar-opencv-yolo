import cv2

gambar = cv2.imread("kucing.jpg")

ukuran = gambar.shape

h, w, c = ukuran

print(ukuran)
print(h * w * c)
print(gambar[100,100])