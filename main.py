import os
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
from picamera2 import Picamera2
from multiprocessing import Process, Queue
from tflite_runtime.interpreter import Interpreter

# Global detection queue
detection_queue = Queue()

# Audio feedback functionality
def speak(text):
    os.system(f'espeak "{text}" --stdout | aplay')

def announce_detection(object_name, distance):
    speak(f"Detected {object_name} at a distance of {distance} centimeters")

# Distance measurement functionality
def calculate_distance(distance_queue):
    GPIO.setmode(GPIO.BCM)
    TRIG = 23
    ECHO = 24
    
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    
    while True:
        GPIO.output(TRIG, False)
        time.sleep(0.5)
        
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        
        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()
        
        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()
        
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        distance = round(distance, 2)

        distance_queue.put(distance)  # Send the calculated distance to the queue

# Object detection functionality
def run_object_detection(frame_queue):
    global detection_queue  # Declare the global queue

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

    print("Running object detection...")

    while True:
        # Wait for a new frame from the frame queue
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
    frame_queue = Queue()

    # Start distance measurement process
    distance_process = Process(target=calculate_distance, args=(distance_queue,))
    distance_process.start()

    # Initialize video stream
    videostream = VideoStream(resolution=(640, 480), framerate=30).start()

    # Start object detection process
    detection_process = Process(target=run_object_detection, args=(frame_queue,))
    detection_process.start()

    try:
        while True:
            frame = videostream.read()
            frame_queue.put(frame)  # Send the frame to the object detection process

            if not detection_queue.empty() and not distance_queue.empty():
                object_name = detection_queue.get()
                distance = distance_queue.get()
                announce_detection(object_name, distance)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        videostream.stop()
        distance_process.join()
        detection_process.join()

if __name__ == "__main__":
    run_vision_tool()
