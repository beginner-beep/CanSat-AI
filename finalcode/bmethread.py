import smbus2
import bme280
import time
from antennaqueue import q
def bmethread(name):
    
    address = 0x76

    bus = smbus2.SMBus(1)

    calibration_params = bme280.load_calibration_params(bus, address)

    while True:
        data = bme280.sample(bus, address, calibration_params)

        temperature_celsius = data.temperature
        humidity = data.humidity
        pressure = data.pressure
        unix_timestamp = int(time.time())
        hours = (unix_timestamp//3600) %24
        minutes = (unix_timestamp%3600)//60
        seconds = unix_timestamp % 60
        time_send = f"{hours:02}:{minutes:02}:{seconds:02}"
        q.put(f"B:{temperature_celsius:.1f},{humidity:.1f},{pressure:.1f},{unix_timestamp}")
        time.sleep(1)

