from multiprocessing import Process, Queue

def run_vision_tool():
    detection_queue = Queue()
    distance_queue = Queue()

    # Start object detection process
    detection_process = Process(target=run_object_detection, args=(detection_queue,))
    detection_process.start()

    # Start distance measurement process
    distance_process = Process(target=calculate_distance, args=(distance_queue,))
    distance_process.start()

    while True:
        if not detection_queue.empty() and not distance_queue.empty():
            object_name = detection_queue.get()
            distance = distance_queue.get()
            announce_detection(object_name, distance)

if __name__ == "__main__":
    run_vision_tool()
