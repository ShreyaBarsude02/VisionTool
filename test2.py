import multiprocessing
import os
import time
from picamera2 import Picamera2
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import pyttsx3
import RPi.GPIO as GPIO

TRIG_PIN = 11
ECHO_PIN = 18

def object_detection(shared_data):
    os.sched_setaffinity(0, {0})
    print("Object detection process running on core 0")

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

    try:
        while True:
            frame = picam2.capture_array()
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
            
            max_confidence = 0
            best_object = None
            for i in range(len(scores)):
                if scores[i] > max_confidence and scores[i] > min_conf_threshold:
                    max_confidence = scores[i]
                    best_object = labels[int(classes[i])]
            
            if best_object:
                print(f"Detected: {best_object} with confidence {max_confidence:.2f}")
                shared_data["object_name"] = best_object
            else:
                shared_data["object_name"] = None

            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping object detection")
    finally:
        picam2.stop()
        picam2.close()

def audio_feedback(shared_data):
    os.sched_setaffinity(0, {1})
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    try:
        while True:
            object_name = shared_data["object_name"]
            distance = shared_data["distance"]
            if object_name:
                message = f"Detected {object_name} at a distance of {distance} centimeters"
                print(f"Audio feedback: {message}")
                engine.say(message)
                engine.runAndWait()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping audio feedback")
    finally:
        engine.stop()

def ultrasonic_sensor(shared_data):
    os.sched_setaffinity(0, {2})
    print("Ultrasonic sensor process running on core 2")

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)

    try:
        while True:
            GPIO.output(TRIG_PIN, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(TRIG_PIN, GPIO.LOW)
            
            while GPIO.input(ECHO_PIN) == GPIO.LOW:
                pulse_start = time.time()
            
            while GPIO.input(ECHO_PIN) == GPIO.HIGH:
                pulse_end = time.time()
            
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150 
            distance = round(distance, 2)
            print(f"Ultrasonic sensor: Distance = {distance} cm")
            shared_data["distance"] = distance
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping ultrasonic sensor")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_data = manager.dict()
    shared_data["object_name"] = None
    shared_data["distance"] = 0

    detection = multiprocessing.Process(target=object_detection, args=(shared_data,))
    ultrasonic = multiprocessing.Process(target=ultrasonic_sensor, args=(shared_data,))
    audio = multiprocessing.Process(target=audio_feedback, args=(shared_data,))

    detection.start()
    ultrasonic.start()
    audio.start()

    detection.join()
    ultrasonic.join()
    audio.join()
