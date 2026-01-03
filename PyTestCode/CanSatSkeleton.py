import multiprocessing as mp
from multiprocessing import Queue,ProcessError
import time
import threading

def computerVision(name):
    while True:
        print(f"Process {name} is running computer vision tasks.")
        time.sleep(2)
        
def data(name):
    while True:
        print(f"Process {name} is handling data tasks.")
        time.sleep(2)

def main():
    print("Starting")

    computer_vision_process= mp.Process(target=computerVision, args=("computer vision",))
    data_process = mp.Process(target=data, args=("other",))

    computer_vision_process.start()
    data_process.start()

    computer_vision_process.join()
    data_process.join()
    

if __name__ == "__main__":
    mp.set_start_method("spawn")  # REQUIRED on Raspberry Pi
    main()