import cv2 as cv2
import numpy as np
import sys
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
images = load_images_from_folder("TestImages")
resImages = []
for image in images:
    resImages.append(image[0:1400, 0:1900])
    cv2.imshow("Main", image)
    cv2.waitKey()
images = resImages

lower_green = np.array([0,0, 150])
upper_green = np.array([255, 255, 255])
##(hMin = 112 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255)
lower_purple = np.array([113,0, 0])
upper_purple = np.array([179, 255, 255])

resultForDisplay = []
displayDifference = []
for image in images:
    
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    greenmask = cv2.inRange(hsvImage, lower_green, upper_green)
    purplemask =cv2.inRange(hsvImage, lower_purple, upper_purple)
    resgreen = cv2.bitwise_and(image, image, mask=greenmask)
    respurple = cv2.bitwise_and(image, image, mask=purplemask)
    finalRes = cv2.bitwise_or(resgreen,respurple)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(finalRes,kernel,iterations = 1)

    displayDifference.extend([image,resgreen,respurple,finalRes,erosion])
   
    cv2.imshow("Main", resgreen)
   ## cv2.waitKey()
    cv2.imshow("Main", respurple)
    ##cv2.waitKey()
    cv2.imshow("Main", finalRes)
    ##cv2.waitKey()
    cv2.imshow("Main" ,erosion)
    cv2.waitKey()
    resultForDisplay.append(finalRes)

im_v = np.concatenate((resultForDisplay[0], resultForDisplay[1]), axis = 1)
new_width= 300
new_height = 300
for i, image in enumerate(displayDifference):
    displayDifference[i] = cv2.resize(image, (new_width, new_height))
imdif = np.concatenate((displayDifference[0],displayDifference[1],displayDifference[2],displayDifference[3],displayDifference[4]), axis = 1)

imdif1 = np.concatenate((displayDifference[5],displayDifference[6],displayDifference[7],displayDifference[8],displayDifference[9]), axis = 1)

cv2.imshow("window1",imdif)
cv2.imshow("window2",imdif1)
cv2.waitKey()

"""cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Resized_Window", 4000, 4000)

cv2.imshow("Resized_Window", im_v)

cv2.waitKey() """