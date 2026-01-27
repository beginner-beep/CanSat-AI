import serial
import time

ser = serial.Serial(
    port="COM5",     # Replace with your COM port
    baudrate=9600,   # Must match LoRa module
    timeout=1
)

# Example: send some hex data
data_to_send = "hello"
ser.write(data_to_send.encode())

time.sleep(0.2)
ser.close()
