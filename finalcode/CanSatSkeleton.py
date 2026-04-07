import multiprocessing as mp
from multiprocessing import Queue,ProcessError, Pipe
import queue
import time
import threading
from gpsthread import gpsthread
from bmethread import bmethread
from antennathread import antennathread
from antennaqueue import q
from computervisionthread import ComputerVision
from processqueue import mpq 
def manager(name,mpq):
    print("[name] Starting")
	
    stop_event = threading.Event()

    threads = [
        threading.Thread(target=gpsthread, args=(stop_event,), daemon=True),
        threading.Thread(target=bmethread, args=(stop_event,), daemon=True),
        threading.Thread(target=antennathread, args=(stop_event, mpq), daemon=True)
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
    computer_vision_process= mp.Process(target=ComputerVision, args=("computer vision", mpq))
    data_process = mp.Process(target=manager, args=("other", mpq))

    computer_vision_process.start()
    data_process.start()

    computer_vision_process.join()
    data_process.join()
    
if __name__ == "__main__":
 
    main()
