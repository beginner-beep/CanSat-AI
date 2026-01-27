import serial

# Raspberry Pi default UART: /dev/serial0
# Laptop: replace with your COM port, e.g. "COM5"
PORT = "/dev/serial0"
BAUD = 9600         # must match your Ebyte module UART baud rate

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Listening for LoRa messages...")

while True:
    data = ser.readline()
    if data:
        print("Received:", data.decode(errors='ignore').strip())
