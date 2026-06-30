from ultralytics import YOLO

deteksi = YOLO("best.pt")

results = deteksi("gambar.jpg")

# results[0].show()

for box in results[0].boxes:
    conf = float(box.conf)
    print(conf)