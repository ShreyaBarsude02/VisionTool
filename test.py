import multiprocessing
import os
import time
from picamera2 import Picamera2
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import pyttsx3
from queue import Queue

TRIG_PIN = 11
ECHO_PIN = 18

def object_detection(detection_queue):
    os.sched_setaffinity(0, {0})
    print("Object detection process running on core 0")

    # Initialize the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
    picam2.start()

    MODEL_NAME = 'model'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    min_conf_threshold = 0.5
    imW, imH = 640, 480

    PATH_TO_CKPT = os.path.join(MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(MODEL_NAME, LABELMAP_NAME)
    
    # Load label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if labels[0] == '???':
        del(labels[0])

    # Load TFLite model and allocate tensors
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    stopped = False

    try:
        while not stopped:
            # Capture frame
            frame = picam2.capture_array()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)
            
            # Preprocess and predict
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Extract detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
            
            # Check detections
            for i in range(len(scores)):
                if (scores[i] > min_conf_threshold):
                    object_name = labels[int(classes[i])]
                    print(f"Detected: {object_name} with confidence {scores[i]:.2f}")
                    detection_queue.put((object_name, None))  # Send object name and distance to queue

            time.sleep(1)  # Sleep to reduce processing load
    except KeyboardInterrupt:
        stopped = True
        picam2.stop()
        print("Camera process stopped.")
    finally:
        picam2.stop()
        picam2.close()

def audio_feedback(detection_queue):
    os.sched_setaffinity(0, {1})
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    while True:
        if not detection_queue.empty():
            object_name, distance = detection_queue.get()
            message = f"Detected {object_name} at a distance of {distance} centimeters"
            print(f"Audio feedback: {message}") 
            engine.say(message)
            engine.runAndWait()
        else:
            break
            
    engine.stop()

def ultrasonic_sensor(detection_queue):
    os.sched_setaffinity(0, {2})  # Run on core 2
    print("Ultrasonic sensor process running on core 2")

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)

    while True:
        GPIO.output(TRIG_PIN, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, GPIO.LOW)
        
        # Wait for echo
        while GPIO.input(ECHO_PIN) == GPIO.LOW:
            pulse_start = time.time()
        
        while GPIO.input(ECHO_PIN) == GPIO.HIGH:
            pulse_end = time.time()
        
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150 
        distance = round(distance, 2)
        print(f"Ultrasonic sensor: Distance = {distance} cm")
        detection_queue.put((None, distance))
        time.sleep(1)

if __name__ == "__main__":
    detection_queue = multiprocessing.Queue()

    detection = multiprocessing.Process(target=object_detection, args=(detection_queue,))
    
    
    detection.start()
    time.sleep(2)

    ultrasonic = multiprocessing.Process(target=ultrasonic_sensor, args=(detection_queue,))

    ultrasonic.start()
    time.sleep(2)


    audio = multiprocessing.Process(target=audio_feedback, args=(detection_queue,))
    audio.start()

    detection.join()
    ultrasonic.join()
    audio.join()

