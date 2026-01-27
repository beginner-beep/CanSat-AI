import serial
import time

ser = serial.Serial(
    port="COM4",     # Replace with your COM port
    baudrate=9600,   # Must match LoRa module
    timeout=1
)

# Example: send some hex data
while True:
    data_to_send = "hello"
    ser.write(data_to_send.encode())
    print("writing")
    time.sleep(1)
ser.close()