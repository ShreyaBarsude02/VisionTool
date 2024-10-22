from multiprocessing import Process, Queue
from object_detection import run_object_detection
from distance_sensor import calculate_distance
from audio_communication import audio_feedback

def main():
    # Create queues to share data between processes
    detection_queue = Queue()
    distance_queue = Queue()
    
    # Process 1: Object detection (Core 1)
    p1 = Process(target=run_object_detection, args=(detection_queue,))
    
    # Process 2: Distance calculation (Core 2)
    p2 = Process(target=calculate_distance, args=(distance_queue,))
    
    # Process 3: Audio feedback (Core 3)
    p3 = Process(target=audio_feedback, args=(detection_queue, distance_queue))
    
    # Start all processes
    p1.start()
    p2.start()
    p3.start()
    
    # Wait for all processes to complete
    p1.join()
    p2.join()
    p3.join()

if __name__ == "__main__":
    main()
