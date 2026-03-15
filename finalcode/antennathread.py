import serial
import time
import numpy as np
from gpiozero import DigitalOutputDevice
from antennaqueue import q

def antennathread(name,mpq):
	
	pin1 = DigitalOutputDevice(18)
	pin2 = DigitalOutputDevice(23)
	pin1.off()
	pin2.off()
	ser = serial.Serial('/dev/serial0', 9600, timeout=1)
	number=0
	time.sleep(2)
	
	while True:
		number +=1
		msg = f"{q.get()}\n"
		approx = mpq.get()
		if approx is not None:
			msg1 = 'C:' + ','.join(f"{int(p[0])},{int(p[1])}" for p in approx[:,0,:]) + '\n'
		else:
			msg1 = ''
		ser.write(msg.encode())
		if msg1:
			ser.write(msg1.encode())
		print(msg, end='')
		time.sleep(1)
