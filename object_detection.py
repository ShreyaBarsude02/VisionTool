import cv2
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
import numpy as np
import os

def run_object_detection():
    MODEL_NAME = 'model'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    min_conf_threshold = 0.8  # Set minimum confidence to 80%
    imW, imH = 640, 480
    
    PATH_TO_CKPT = os.path.join(MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(MODEL_NAME, LABELMAP_NAME)
    
    # Load labels
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if labels[0] == '???':
        del(labels[0])
    
    # Load the model
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    # Initialize the video stream
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()

    print("Running object detection, press Ctrl+C to stop...")

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
            
            # Get detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
            classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Detected class indexes
            scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
            
            # Draw boxes and labels for confidence > 80%
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    object_name = labels[int(classes[i])]  # Lookup object name from label file
                    confidence = int(scores[i] * 100)
                    print(f"Detected: {object_name} with confidence: {scores[i]:.2f}")

                    # Get bounding box coordinates
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    # Draw the bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                    # Draw label and confidence score
                    label = f'{object_name}: {confidence}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Show the frame with detection
            cv2.imshow('Object Detector', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Stopped by user.")
    
    finally:
        # Clean up
        cv2.destroyAllWindows()
        videostream.stop()

# Helper class for video stream
class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"size": resolution, "format": "RGB888"}))
        self.picam2.start()
        self.frame = None

    def start(self):
        return self

    def read(self):
        return self.picam2.capture_array()

    def stop(self):
        self.picam2.stop()

# Run the detection
if __name__ == "__main__":
    run_object_detection()
