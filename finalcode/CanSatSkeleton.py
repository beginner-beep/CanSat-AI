import multiprocessing as mp
from multiprocessing import Queue,ProcessError
import time
import threading
from gpsthread import gpsthread
from bmethread import bmethread

def manager(name):
    print("[name] Starting")

    stop_event = threading.Event()

    threads = [
        threading.Thread(target=gpsthread, args=(stop_event,), daemon=True),
        threading.Thread(target=bmethread, args=(stop_event,), daemon=True),
    ]

    for t in threads:
        t.start()

    try:
        while True:
            print("running")
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
def main():
    print("Starting")

  #  computer_vision_process= mp.Process(target=computerVision, args=("computer vision",))
    data_process = mp.Process(target=manager, args=("other",))

   # computer_vision_process.start()
    data_process.start()

    #computer_vision_process.join()
    data_process.join()
    
    
    

if __name__ == "__main__":
    mp.set_start_method("spawn")  # REQUIRED on Raspberry Pi
    main()