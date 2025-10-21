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

for i, image in enumerate(images):
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

"""cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Resized_Window", 4000, 4000)

cv2.imshow("Resized_Window", im_v)

cv2.waitKey() """