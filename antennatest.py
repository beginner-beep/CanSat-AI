import serial
import time
from gpiozero import DigitalOutputDevice
pin1 = DigitalOutputDevice(18)
pin2 = DigitalOutputDevice(23)
pin1.off()
pin2.off()
ser = serial.Serial('/dev/serial0', 9600, timeout=1)
number=0
time.sleep(2)
while True:
    number +=1
    msg = f"hello {number}"
    ser.write(msg.encode())
    print(msg, end=' ')
    time.sleep(1)
