import pyttsx3

def audio_feedback(detection_queue, distance_queue):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    while True:
        if not detection_queue.empty() and not distance_queue.empty():
            detected_object = detection_queue.get()
            distance = distance_queue.get()
            message = f"The closest object is a {detected_object} at a distance of {distance} centimeters."
            engine.say(message)
            engine.runAndWait()
