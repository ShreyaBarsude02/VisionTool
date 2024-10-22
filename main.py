import os
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
from picamera2 import Picamera2
from multiprocessing import Process, Queue
from threading import Thread
from tflite_runtime.interpreter import Interpreter

# Global queues
detection_queue = Queue()
frame_queue = Queue()

# Audio feedback functionality
def speak(text):
    os.system(f'espeak "{text}" --stdout | aplay')

def announce_detection(object_name, distance):
    speak(f"Detected {object_name} at a distance of {distance} centimeters")

# Distance measurement functionality
def calculate_distance(distance_queue):
    distance = 0
    while True:
        distance = distance + 1  # Simulate distance increase for now
        distance_queue.put(distance)  # Send the calculated distance to the queue
        time.sleep(1)  # Simulate delay for testing

# Object detection functionality
def run_object_detection(frame_queue):
    global detection_queue  # Declare the global queue

    MODEL_NAME = 'model'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    min_conf_threshold = 0.5  # Confidence threshold

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

    print("Running object detection...")

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
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
                if scores[i] > min_conf_threshold:
                    object_name = labels[int(classes[i])]
                    print(f"Detected: {object_name} with confidence {scores[i]:.2f}")
                    detection_queue.put(object_name)  # Send detected object to the global queue

# VideoStream class for camera
class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.picam2 = Picamera2()
        try:
            self.picam2.configure(self.picam2.create_preview_configuration(main={"size": resolution, "format": "RGB888"}))
            self.picam2.start()
        except RuntimeError as e:
            print(f"Camera initialization failed: {e}")
            raise
    
    def start(self):
        return self

    def read(self):
        return self.picam2.capture_array()

    def stop(self):
        self.picam2.stop()

# Main function to run the vision tool
def run_vision_tool():
    distance_queue = Queue()

    # Start distance measurement process
    distance_process = Process(target=calculate_distance, args=(distance_queue,))
    distance_process.start()

    # Initialize video stream
    videostream = VideoStream(resolution=(640, 480), framerate=30).start()

    # Start object detection thread
    detection_thread = Thread(target=run_object_detection, args=(frame_queue,))
    detection_thread.start()

    try:
        while True:
            # Read frame from the camera
            frame = videostream.read()
            
            # Put the frame into the frame_queue for object detection
            if not frame_queue.full():
                frame_queue.put(frame)

            # Process detection and distance data
            if not detection_queue.empty() and not distance_queue.empty():
                object_name = detection_queue.get()
                distance = distance_queue.get()
                announce_detection(object_name, distance)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        videostream.stop()
        distance_process.join()
        detection_thread.join()

if __name__ == "__main__":
    run_vision_tool()
