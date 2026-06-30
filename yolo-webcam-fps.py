import cv2
import time
from ultralytics import YOLO

sumber = 1  # index webcam
video = cv2.VideoCapture(sumber)

deteksi = YOLO("best.pt")

# Inisialisasi waktu untuk perhitungan FPS
prev_time = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("Gagal mengambil gambar dari webcam.")
        break

    # Jalankan deteksi YOLO
    results = deteksi(frame)
    box = results[0].plot()

    # --- Menghitung FPS ---
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # --- Menampilkan FPS di Frame ---
    # format teks FPS menjadi string tanpa desimal
    fps_text = f"FPS: {int(fps)}"
    
    # Menulis teks FPS ke gambar (koordinat x=10, y=30, warna hijau (0, 255, 0), ketebalan=2)
    cv2.putText(
        box, 
        fps_text, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2, 
        cv2.LINE_AA
    )

    # Tampilkan hasil
    cv2.imshow("Webcam", box)

    if cv2.waitKey(1) == 27:  # tombol esc
        break

video.release()
cv2.destroyAllWindows()