import cv2
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

def open_webcam():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap

def open_file(path):
    return cv2.VideoCapture(path)

def read_image(path):
    return cv2.imread(path)

def release(cap):
    cap.release()
    cv2.destroyAllWindows()