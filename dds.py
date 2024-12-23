import cv2
import mediapipe as mp
from playsound import playsound
import time
import threading

# Initialize MediaPipe FaceMesh and Drawing Utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define landmarks for eyes (right and left eye) and mouth (for yawning detection)
RIGHT_EYE = [33, 160, 158, 133, 153, 144, 163, 7, 163, 133, 153]
LEFT_EYE = [362, 385, 387, 263, 373, 380, 389, 249, 362, 263, 373]
MOUTH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]  # Check landmark indices

# Eye aspect ratio threshold to detect closed eyes
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 48  # Number of consecutive frames the eyes need to be closed

# Yawning detection threshold
YAWN_THRESH = 35  # Threshold for mouth aspect ratio to detect yawning

# Initialize variables
eye_counter = 0
yawn_counter = 0
ALERT_PLAYED = False

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = abs(eye[1].y - eye[5].y)
    B = abs(eye[2].x - eye[4].x)
    C = abs(eye[0].x - eye[3].x)
    return (A + B) / (2.0 * C)

# Function to calculate the mouth aspect ratio (MAR) for yawning detection
def mouth_aspect_ratio(mouth):
    try:
        # Ensure valid indices by printing mouth landmarks
        for landmark in mouth:
            print(f"Landmark ({landmark.x}, {landmark.y})")

        A = abs(mouth[2].y - mouth[10].y)
        B = abs(mouth[4].x - mouth[8].x)
        return A / B
    except IndexError as e:
        print("Index error:", e)
        return 0

# Function to play an alert sound
def play_alert():
    try:
        playsound('alert.mp3')  # Ensure alert.mp3 is in the same directory as the script
    except Exception as e:
        print(f"Error playing sound: {e}")

# Function to reset alert state after sound completes
def reset_alert_state():
    global ALERT_PLAYED
    ALERT_PLAYED = False

# Initialize webcam
cap = cv2.VideoCapture(0)

# Start the FaceMesh detector
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect faces and landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract the eye landmarks for both eyes and the mouth landmarks for yawning detection
                right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE]
                left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE]
                mouth = [face_landmarks.landmark[i] for i in MOUTH]

                # Calculate the Eye Aspect Ratio (EAR) for both eyes
                right_ear = eye_aspect_ratio(right_eye)
                left_ear = eye_aspect_ratio(left_eye)

                # Calculate the Mouth Aspect Ratio (MAR) for yawning
                mar = mouth_aspect_ratio(mouth)

                # Detect eye closure
                if right_ear < EYE_AR_THRESH and left_ear < EYE_AR_THRESH:
                    eye_counter += 1
                else:
                    eye_counter = 0

                # Detect yawning
                if mar > YAWN_THRESH:
                    yawn_counter += 1
                else:
                    yawn_counter = 0

                # If the eyes have been closed for a certain number of frames, play the alert sound
                if eye_counter >= EYE_AR_CONSEC_FRAMES and not ALERT_PLAYED:
                    print("Eyes closed detected!")
                    ALERT_PLAYED = True  # Prevent playing the sound again before resetting
                    threading.Thread(target=play_alert).start()  # Play the alert sound in a separate thread
                    threading.Thread(target=reset_alert_state).start()  # Reset the alert state after sound completes

                # If yawning is detected, play the alert sound
                if yawn_counter >= 10 and not ALERT_PLAYED:
                    print("Yawning detected!")
                    ALERT_PLAYED = True  # Prevent playing the sound again before resetting
                    threading.Thread(target=play_alert).start()  # Play the alert sound in a separate thread
                    threading.Thread(target=reset_alert_state).start()  # Reset the alert state after sound completes

        # Display the frame
        cv2.imshow('Yawning and Eye Closure Detection', frame)

        # Stop the alert and exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting and stopping alert...")
            ALERT_PLAYED = False  # Reset the alert when 'q' is pressed
            break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
