import cv2
import time
from src.pose import get_pose_model, get_landmarks, draw_landmarks
from src.angles import get_all_angles
from src.analyzer import analyze_posture
from src.feedback import draw_angles, draw_status
from src.capture import open_webcam, release
import mediapipe as mp

def run_live():
    cap = open_webcam()
    pose = get_pose_model(static_image_mode=False)
    VisionRunningMode = mp.tasks.vision.RunningMode

    print("Live mode running — press Q to quit")
    timestamp = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp += 1
        results = pose.detect_for_video(mp_image, timestamp)

        frame = draw_landmarks(frame, results)
        landmarks = get_landmarks(results)

        if landmarks:
            angles   = get_all_angles(landmarks)
            analysis = analyze_posture(angles)
            frame    = draw_angles(frame, angles)
            frame    = draw_status(frame, analysis)

        # cv2.imshow("Live Posture Monitor", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

    release(cap)
    pose.close()
    cv2.destroyAllWindows()