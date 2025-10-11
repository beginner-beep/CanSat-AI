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

for image in images:
    cv2.imshow("Main", image)
    cv2.waitKey()
lower_green = np.array([0,0, 150])
upper_green = np.array([255, 255, 255])
##(hMin = 112 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255)
lower_purple = np.array([113,0, 0])
upper_purple = np.array([179, 255, 255])

resultForDisplay = []

for image in images:
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    greenmask = cv2.inRange(hsvImage, lower_green, upper_green)
    purplemask =cv2.inRange(hsvImage, lower_purple, upper_purple)
    resgreen = cv2.bitwise_and(image, image, mask=greenmask)
    respurple = cv2.bitwise_and(image, image, mask=purplemask)
    finalRes = cv2.bitwise_or(resgreen,respurple)
    cv2.imshow("Main", resgreen)
    cv2.waitKey()
    cv2.imshow("Main", respurple)
    cv2.waitKey()
    cv2.imshow("Main", finalRes)
    cv2.waitKey()
    resultForDisplay.append(finalRes)

numpy_horizontal_concat = np.concatenate((finalRes), axis=1)

cv2.imshow('Main', numpy_horizontal_concat)
cv2.imshow('Numpy Vertical', numpy_horizontal_concat)
cv2.waitKey()