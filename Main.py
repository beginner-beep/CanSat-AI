import cv2 as cv2
import numpy as np
import sys
import os

img2 = cv2.imread(cv2.samples.findFile("TestImages/Screenshot 2025-10-11 093736.png"))
img = cv2.imread(cv2.samples.findFile("TestImages/Screenshot 2025-10-08 102026.png"))

if img is None:
    sys.exit("Could not read the image.")

if img2 is None:  
     sys.exit("Could not read the image.")


hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsvImage2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

lower_green = np.array([0,0, 150])
upper_green = np.array([255, 255, 255])
##(hMin = 112 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255)
lower_purple = np.array([113,0, 0])
upper_purple = np.array([179, 255, 255])
mask = cv2.inRange(hsvImage, lower_green, upper_green)
res = cv2.bitwise_and(img, img, mask=mask)
mask2 =cv2.inRange(hsvImage2, lower_purple, upper_purple)
res1 = cv2.bitwise_and(img2, img2, mask=mask2)
print(mask)
hsv_frame = cv2.cvtColor(hsvImage, cv2.COLOR_BGR2HSV)

cv2.imshow("Display window", res)
k = cv2.waitKey(0)
cv2.imshow("Display window", res1)
k = cv2.waitKey(0)

