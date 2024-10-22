import multiprocessing
import os
import time
from picamera2 import Picamera2
from threading import Thread
from queue import Queue 
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

def object_detection():
    os.sched_setaffinity(0, {0})
    print("object detection process running on core 0")

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

    stopped = False

    try:
        while not stopped:
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
            
            for i in range(len(scores)):
                if (scores[i] > min_conf_threshold):
                    object_name = labels[int(classes[i])]
                    print(f"Detected: {object_name} with confidence {scores[i]:.2f}")

            time.sleep(1) 
    except KeyboardInterrupt:
        stopped = True
        picam2.stop()
        print("Camera process stopped.")

if __name__ == "__main__":
    
    detection = multiprocessing.Process(target=object_detection)
    detection.start()

    detection.join()
