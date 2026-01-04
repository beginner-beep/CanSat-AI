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
	##(hMin = 112 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255)
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
	   
		min_area = 500  # ignore tiny noise
		filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
		filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
		image_copy = thresh.copy()
	# Keep only the largest N zones (e.g., 3)
		contours = filtered_contours[:3]
		
		cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
		
		scale = 0.5
		#cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
		image_copy = cv2.resize(image_copy, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		return image, image_copy
		
	   
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
		image = picam2.capture_array()
		cv2.imshow('test', image)
		# if frame is read correctly ret is True
		#if not image:
		 #   print("Can't receive frame (stream end?). Exiting ...")
		  #  break
		
		image, image_copy = ApplyFilters(image)
	 
		
		if cv2.waitKey(10) == ord('q'):
			break
		cv2.imshow("erosion", image)
		cv2.imshow("frame", image_copy)
		mpq.put("hello")
	cap.release()
	cv2.destroyAllWindows()


	"""for i, image in enumerate(images):
		images[i] = (image[0:1400, 0:1900])

	##colour filters: select values with colour picker script
	lower_green = np.array([0,0, 99])
	upper_green = np.array([255, 255, 255])
	##(hMin = 112 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255)
	lower_purple = np.array([113,0, 0])
	upper_purple = np.array([179, 255, 255])

	resultForDisplay = []
	displayDifference = []
	contours = []
	for image in images:
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
	  
		#erosion = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernelClosing, iterations=1)   
	   # cv2.imshow("Erosion", erosion)
		
		#cv2.waitKey()
		
		displayDifference.extend([image,resgreen,respurple,finalRes,erosion])
		
		ret,thresh = cv2.threshold(erosion, 10,10,10, cv2.THRESH_BINARY)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelClosing, iterations=3)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	   # cv2.imshow("Main", resgreen)
	   
		image_copy = thresh.copy()
		
		cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
	   # cv2.imshow("Main", finalRes)
		scale = 0.2
	  
		img1_small = cv2.resize(erosion, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		img3_small = cv2.resize(image_copy, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		finalRes = cv2.resize(finalRes, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		
	# Combine side by side
		comparisonErosion = cv2.hconcat([finalRes, img1_small, img3_small])
	   
		cv2.imshow("Comparison", comparisonErosion)

	  #  cv2.imshow('contours_none_image1.jpg', image_copy)

		cv2.waitKey()
		resultForDisplay.append(finalRes)

	im_v = np.concatenate((resultForDisplay[0], resultForDisplay[1]), axis = 1)
	new_width= 300
	new_height = 300
	for i, image in enumerate(displayDifference):
		displayDifference[i] = cv2.resize(image, (new_width, new_height))

	for i, step in enumerate(displayDifference):
		cv2.imshow("window", np.concatenate((displayDifference[0+i*5],displayDifference[1+ i *5],displayDifference[2+i*5],displayDifference[3+ i*5],displayDifference[4+i*5]), axis = 1))
		cv2.waitKey()
	cv2.waitKey()

	"""
