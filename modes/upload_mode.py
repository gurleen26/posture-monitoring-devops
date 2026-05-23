import cv2
import os
import mediapipe as mp
from src.pose import get_pose_model, get_landmarks, draw_landmarks
from src.angles import get_all_angles
from src.analyzer import analyze_posture
from src.feedback import draw_angles, draw_status
from src.capture import read_image, open_file, release
from config import OUTPUT_IMAGES, OUTPUT_VIDEOS

def resize_to_fit(frame, max_width=900, max_height=700):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))

def run_upload(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        run_on_image(path)
    elif ext in [".mp4", ".mov", ".avi"]:
        run_on_video(path)
    else:
        print("Unsupported file format.")

def run_on_image(path):
    pose  = get_pose_model(static_image_mode=True)
    frame = read_image(path)

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results   = pose.detect(mp_image)
    frame     = draw_landmarks(frame, results)
    landmarks = get_landmarks(results)

    if landmarks:
        angles   = get_all_angles(landmarks)
        analysis = analyze_posture(angles)
        frame    = draw_angles(frame, angles)
        frame    = draw_status(frame, analysis)
        print("Angles  :", angles)
        print("Analysis:", analysis)
    else:
        print("No person detected in image.")

    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    out_path = os.path.join(OUTPUT_IMAGES, os.path.basename(path))
    cv2.imwrite(out_path, frame)
    print(f"Saved to {out_path}")

    frame = resize_to_fit(frame)
    cv2.namedWindow("Upload Analysis", cv2.WINDOW_NORMAL)
    # cv2.imshow("Upload Analysis", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pose.close()

def run_on_video(path):
    pose = get_pose_model(static_image_mode=False)
    cap  = open_file(path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get original dimensions
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    print(f"Video: {orig_w}x{orig_h} @ {fps}fps")

    # Cap display and save resolution at 1280 wide max
    MAX_W = 1280
    if orig_w > MAX_W:
        scale = MAX_W / orig_w
        save_w = MAX_W
        save_h = int(orig_h * scale)
    else:
        save_w = orig_w
        save_h = orig_h

    print(f"Processing at: {save_w}x{save_h}")

    os.makedirs(OUTPUT_VIDEOS, exist_ok=True)
    out_path = os.path.join(OUTPUT_VIDEOS, "annotated_" + os.path.basename(path))
    writer   = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (save_w, save_h)
    )

    cv2.namedWindow("Upload Video Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Upload Video Analysis", min(save_w, 900), min(save_h, 700))

    timestamp  = 0
    frame_skip = 2  # process every 2nd frame for speed on large videos

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp += 1

        # Resize early — before MediaPipe processing
        frame = cv2.resize(frame, (save_w, save_h))

        # Skip frames on large videos to avoid lag
        if timestamp % frame_skip != 0:
            writer.write(frame)
            # cv2.imshow("Upload Video Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = pose.detect_for_video(mp_image, timestamp)
        frame    = draw_landmarks(frame, results)
        landmarks = get_landmarks(results)

        if landmarks:
            angles   = get_all_angles(landmarks)
            analysis = analyze_posture(angles)
            frame    = draw_angles(frame, angles)
            frame    = draw_status(frame, analysis)

        writer.write(frame)
        # cv2.imshow("Upload Video Analysis", frame)

        # waitKey based on fps to play at correct speed
        delay = max(1, int(1000 / fps))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    print(f"Saved to {out_path}")
    release(cap)
    writer.release()
    pose.close()
    cv2.destroyAllWindows()