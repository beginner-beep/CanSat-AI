import serial
import time 
import string
import pynmea2
from antennaqueue import q

def gpsthread(name):
    
    ser=serial.Serial("/dev/ttyAMA3", baudrate=38400, timeout=1)
    alt = None
    lat = None
    lng = None
    
    while True:

        dataout =pynmea2.NMEAStreamReader() 
        newdata=ser.readline()
        if b'$GNGGA' in newdata:
            decoded = newdata.decode('utf-8', errors='ignore')
            msg = pynmea2.parse(decoded)
            alt = msg.altitude  # meters
            
        if b'$GNGLL' in newdata:
            newmsg=pynmea2.parse(newdata.decode('utf-8'))  
            lat=newmsg.latitude 
            lng=newmsg.longitude 
            
            # Send combined GPS data when we have coordinates
            if alt is not None:
                unix_timestamp = int(time.time())
                gps = f"G:{lat:.5f},{lng:.5f},{alt:.5f},{unix_timestamp}"
                q.put(gps)
