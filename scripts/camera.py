import cv2

def get_image_from_camera():
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        raise Exception("Could not open video device")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    ret, frame = cap.read()
    
    if not ret:
        raise Exception("Failed to capture image")
    
    cap.release()
    
    return frame
