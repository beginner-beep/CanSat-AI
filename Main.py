import cv2 as cv2
import numpy as np
import sys

img = cv2.imread(cv2.samples.findFile("testFoto1.png"))
hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

if img is None:
    sys.exit("Could not read the image.")



lower_green = np.array([0,0, 150])
upper_green = np.array([255, 255, 255])

mask = cv2.inRange(hsvImage, lower_green, upper_green)
res = cv2.bitwise_and(img, img, mask=mask)


print(mask)
hsv_frame = cv2.cvtColor(hsvImage, cv2.COLOR_BGR2HSV)

cv2.imshow("Display window", res)
k = cv2.waitKey(0)

	






