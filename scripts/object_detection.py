import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

# Load the TFLite model
interpreter = tflite.Interpreter(model_path='../model/ssd_mobilenet_v2.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_object_detection(image):
    input_shape = input_details[0]['shape']
    resized_image = cv2.resize(image, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_image, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data,"\n")
    return output_data
