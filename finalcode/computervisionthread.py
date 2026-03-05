import cv2 as cv2
import numpy as np
import sys
import os
from picamera2 import Picamera2

import serial
import time
import board
import busio

def ComputerVision(name,mpq):
	picam2= Picamera2()
	config = picam2.create_preview_configuration(
		main={"size": ( 1920, 1080), "format" : "RGB888"}
	)
	picam2.configure(config)
	picam2.start()

	lower_green = np.array([0,0, 99])
	upper_green = np.array([255, 255, 255])
	##(hMin = 112 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255) volgens de colour picker script
	lower_purple = np.array([113,0, 0])
	upper_purple = np.array([179, 255, 255])

	def ApplyFilters(image):
		hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		greenmask = cv2.inRange(hsvImage, lower_green, upper_green)
		purplemask =cv2.inRange(hsvImage, lower_purple, upper_purple)

		resgreen = cv2.bitwise_and(image, image, mask=greenmask)
		respurple = cv2.bitwise_and(image, image, mask=purplemask)

		finalRes = cv2.bitwise_or(resgreen,respurple)
		finalRes = cv2.cvtColor(finalRes,cv2.COLOR_BGR2GRAY)
		kernel = np.ones((2,2),np.uint8)
		erosion = cv2.erode(finalRes,kernel,iterations = 5)
		kernelClosing = np.ones((8,8), np.uint8)
		
		ret,thresh = cv2.threshold(erosion, 10,10,10, cv2.THRESH_BINARY)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelClosing, iterations=3)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	   # cv2.imshow("Main", resgreen)
	   
		min_area = 500 
		filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
		filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
		image_copy = thresh.copy()
		#keeping only 1, if i want to use more for local possible but this is better for transmitting
		contours = filtered_contours[:1]
		epsilon = 0.01 * cv2.arcLength(contours[0], True)  # tuning parameter
		approx = cv2.approxPolyDP(contours[0], epsilon, True)
		
		cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
		
		scale = 0.5
		#cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
		image_copy = cv2.resize(image_copy, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		return image, image_copy, approx
		
	   
	#def load_videos_from_folder(folder):
	 #   videos = []
	#  for filename in os.listdir(folder):
	  #      vid = cv2.VideoCapture(os.path.join(folder,filename))
	   #     if vid is not None:
		#        videos.append(vid)
		#return videos

	#videos = load_videos_from_folder("TestVideos")

	#cap = videos[1]

	##colour filters: select values with colour v

	while True:
		start = time.time()
           
		image = picam2.capture_array()
		cv2.imshow('test', image)
		# if frame is read correctly ret is True
		#if not image:
		 #   print("Can't receive frame (stream end?). Exiting ...")
		  #  break
		
		image, image_copy, approx = ApplyFilters(image)

		elapsed = time.time()
		time.sleep(max(0, 1.0 - elapsed))
  
		if cv2.waitKey(10) == ord('q'):
			break
		cv2.imshow("erosion", image)
		cv2.imshow("frame", image_copy)
		mpq.put(approx)
	cap.release()
	cv2.destroyAllWindows()


	
