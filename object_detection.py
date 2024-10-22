import cv2
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
import numpy as np
import os
from queue import Queue  # Make sure to import Queue

def run_object_detection(detection_queue):
    MODEL_NAME = 'model'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    min_conf_threshold = 0.5  # Confidence threshold
    imW, imH = 640, 480
    
    # Load model and labels
    PATH_TO_CKPT = os.path.join(MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(MODEL_NAME, LABELMAP_NAME)
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if labels[0] == '???':
        del(labels[0])

    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    # Initialize video stream
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    print("Running object detection...")

    try:
        while True:
            frame = videostream.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)
            
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
            
            for i in range(len(scores)):
                if (scores[i] > min_conf_threshold):
                    object_name = labels[int(classes[i])]
                    detection_queue.put(object_name)  # Send detected object to the queue
    
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        videostream.stop()

# VideoStream class for camera
class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"size": resolution, "format": "RGB888"}))
        self.picam2.start()
    
    def start(self):
        return self

    def read(self):
        return self.picam2.capture_array()

    def stop(self):
        self.picam2.stop()

if __name__ == "__main__":
    run_object_detection()
