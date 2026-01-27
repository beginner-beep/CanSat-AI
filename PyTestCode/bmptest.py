import board
import busio
import adafruit_bmp280
import time

i2c = busio.I2C(board.SCL, board.SDA)
bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(i2c, address=0x76)

bmp280.sea_level_pressure = 1013.25

try:
    
    while True:
        print(f"Temperature:{bmp280.temperature: .2f} celsius")
        print(f"altitude:{bmp280.pressure: .2f} hPa")
        print(f"altitude:{bmp280.altitude: .2f} m")
        print(" ------------------- ")
        time.sleep(1)
except KeyboardInterrupt:
    print("Exciting...")