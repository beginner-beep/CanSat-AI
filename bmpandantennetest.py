import serial
import time
import board
import busio
import bme280
#from picamera2 import Picamera2, Preview

ser = serial.Serial('/dev/serial0', 9600, timeout=1)
number=0

i2c = busio.I2C(board.SCL, board.SDA)
bmp280 = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=0x76)
bmp280.sea_level_pressure = 1013.25

#picam2 = Picamera2()
#camera_config = picam2.create_preview_configuration()
#picam2.configure(camera_config)

#picam2.start_preview(Preview.QTGL)
#picam2.start()
#time.sleep(2)
#picam2.capture_file("test.jpg")
while True:
    number +=1
    print(f"Temperature:{bme280.temperature: .2f} celsius")
    print(f"altitude:{bme280.pressure: .2f} hPa")
    print(f"altitude:{bme280.altitude: .2f} m")
    
    print(" ------------------- ")
    msg = f"Temperature:{bmp280.temperature: .2f} celsius,altitude:{bmp280.pressure: .2f} hPa, altitude:{bmp280.altitude: .2f} m "
    ser.write(msg.encode())
    print(msg, end=' ')
    time.sleep(1)