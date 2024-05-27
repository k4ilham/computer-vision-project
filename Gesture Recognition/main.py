import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hand
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi Video Capture
cap = cv2.VideoCapture(0)

# Fungsi untuk mengenali gesture berdasarkan posisi landmark
def recognize_gesture(landmarks):
    # Dapatkan koordinat landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # Hitung jarak (menggunakan Euclidean distance atau metode lainnya)
    thumb_index_dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    index_middle_dist = ((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2) ** 0.5
    middle_ring_dist = ((middle_tip.x - ring_tip.x) ** 2 + (middle_tip.y - ring_tip.y) ** 2) ** 0.5
    ring_pinky_dist = ((ring_tip.x - pinky_tip.x) ** 2 + (ring_tip.y - pinky_tip.y) ** 2) ** 0.5

    # Thresholds untuk mendeteksi gesture (misalnya angka 1, 2, 3)
    if thumb_index_dist < 0.05 and index_middle_dist > 0.1 and middle_ring_dist > 0.1 and ring_pinky_dist > 0.1:
        return "Gesture 1"
    elif thumb_index_dist > 0.1 and index_middle_dist < 0.05 and middle_ring_dist > 0.1 and ring_pinky_dist > 0.1:
        return "Gesture 2"
    elif thumb_index_dist > 0.1 and index_middle_dist > 0.1 and middle_ring_dist < 0.05 and ring_pinky_dist > 0.1:
        return "Gesture 3"
    else:
        return "Unknown Gesture"

# Pengaturan MediaPipe
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi warna BGR ke RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Proses deteksi tangan
        results = hands.process(image)

        # Konversi warna RGB ke BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Tampilkan hasil deteksi
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Mengenali gesture
                gesture = recognize_gesture(hand_landmarks.landmark)
                cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Tampilkan frame video
        cv2.imshow('Gesture Recognition', image)

        # Keluar dengan menekan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Lepaskan resources
cap.release()
cv2.destroyAllWindows()
