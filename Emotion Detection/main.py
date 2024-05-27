import cv2
from fer import FER

# Inisialisasi detektor emosi
detector = FER(mtcnn=True)

# Inisialisasi Video Capture
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi emosi
    result = detector.detect_emotions(frame)

    # Tampilkan hasil deteksi
    for r in result:
        (x, y, w, h) = r["box"]
        emotion, score = max(r["emotions"].items(), key=lambda x: x[1])
        
        # Gambar persegi di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Tampilkan emosi di atas persegi
        cv2.putText(frame, f'{emotion}: {score:.2f}', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Tampilkan frame video
    cv2.imshow('Emotion Detection', frame)

    # Keluar dengan menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan resources
cap.release()
cv2.destroyAllWindows()
