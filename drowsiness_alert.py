import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import threading

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks indexes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_landmarks, frame_width, frame_height):
    points = [(int(l.x * frame_width), int(l.y * frame_height)) for l in eye_landmarks]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# Alert function (runs in separate thread)
def play_alert():
    playsound("alarme-342932.mp3")

# EAR threshold and frame limits
EAR_THRESHOLD = 0.25   # below this → eyes closed
CLOSED_FRAMES = 40     # must be closed for 20 frames (~1 sec at 20fps)

frame_count = 0

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            left_eye = [landmarks.landmark[i] for i in LEFT_EYE]
            right_eye = [landmarks.landmark[i] for i in RIGHT_EYE]

            left_ear = eye_aspect_ratio(left_eye, w, h)
            right_ear = eye_aspect_ratio(right_eye, w, h)
            ear = (left_ear + right_ear) / 2.0

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check drowsiness
            if ear < EAR_THRESHOLD:
                frame_count += 1
                if frame_count >= CLOSED_FRAMES:
                    cv2.putText(frame, "DROWSY!", (200, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    # Play sound alert in background
                    threading.Thread(target=play_alert, daemon=True).start()
            else:
                frame_count = 0  # reset if eyes open

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
