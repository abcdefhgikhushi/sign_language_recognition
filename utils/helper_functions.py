import cv2
import numpy as np

def preprocess_image(img, target_size):
    """Resize and normalize image"""
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def draw_hand_rectangle(frame, start_point=(100, 100), end_point=(400, 400)):
    """Draw rectangle for hand region"""
    cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
    return frame