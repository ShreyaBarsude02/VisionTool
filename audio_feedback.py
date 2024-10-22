import os

def speak(text):
    os.system(f'espeak "{text}" --stdout | aplay')

def announce_detection(object_name, distance):
    speak(f"Detected {object_name} at a distance of {distance} centimeters")

if __name__ == "__main__":
    announce_detection("person", 50)
