from multiprocessing import Process
from scripts.object_detection import run_object_detection
from scripts.distance_measurement import measure_distance
from scripts.audio_feedback import speak
from scripts.camera import get_image_from_camera

def run_object_detection_core():
    while True:
        image = get_image_from_camera()
        detected_objects = run_object_detection(image)
        print("Objects detected:", detected_objects)

def run_distance_measurement_core():
    while True:
        distance = measure_distance()
        print("Measured distance:", distance)

def run_audio_output_core():
    while True:
        object = "closest object"
        distance = 20
        speak(f"Detected {object} at {distance} centimeters")

if __name__ == '__main__':
    p1 = Process(target=run_object_detection_core)
    p2 = Process(target=run_distance_measurement_core)
    p3 = Process(target=run_audio_output_core)

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
