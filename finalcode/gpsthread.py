import serial
import time 
import string
import pynmea2
from antennaqueue import q

def gpsthread(name):
    
    ser=serial.Serial("/dev/ttyAMA3", baudrate=38400, timeout=1)
    while True:

        dataout =pynmea2.NMEAStreamReader() 
        newdata=ser.readline()
        if b'$GNGGA' in newdata:
            decoded = newdata.decode('utf-8', errors='ignore')
            msg = pynmea2.parse(decoded)
		
            alt = msg.altitude  # meters
            q.put(f"altitude: {alt}")
          #  print(f"Altitude = {alt} meters")
        if b'$GNGLL' in newdata:
          #  print(newdata.decode('utf-8'))
            newmsg=pynmea2.parse(newdata.decode('utf-8'))  
            lat=newmsg.latitude 
            lng=newmsg.longitude 
            gps = "Latitude=" + str(lat) + "and Longitude=" +str(lng)
            q.put(gps)
           # print(gps)
