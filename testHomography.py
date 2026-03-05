import cv2
import numpy as np
import os
#determine masks with colour picker hsv.py
lower_green = np.array([0,0, 99])
upper_green = np.array([255, 255, 255])

lower_purple = np.array([113,0, 0])
upper_purple = np.array([179, 255, 255])

def plotTrajectory(relative_camera_position):
    print("relative x position:"+ f"{x_pos}")
    print("relative y position:"+ f"{y_pos}")
    print("relative z position:"+ f"{z_pos}")
    
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
   
    min_area = 500  # ignore tiny noise
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
    image_copy = thresh.copy()

    contours = filtered_contours[:3]
    
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    return image, image_copy,contours

def load_videos_from_folder(folder):
    videos = []
    for filename in os.listdir(folder):
        vid = cv2.VideoCapture(os.path.join(folder,filename))
        if vid is not None:
            videos.append(vid)
    return videos

scale = 0.5

relative_camera_position = [0,0,0] 
  
videos = load_videos_from_folder("TestVideos")
cap = videos[1]

feature_detector = cv2.ORB_create(50)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
kp1, des1 = feature_detector.detectAndCompute(prev_gray, None)

H_total = np.eye(3)  # accumulate motion
DebugCounter=0
while True:
   
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    kp2, des2 = feature_detector.detectAndCompute(gray, None)
    matches = bf_matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

  
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0) 
    if H is None:
        continue
   
    H_total = H_total @ H

    h, w = prev_gray.shape
    
    j, zone_pts_ref,contours = ApplyFilters(frame)
    cv2.drawContours(image=gray, contours=contours, contourIdx=-1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    stabilized = cv2.warpPerspective(gray, H_total, (w, h))
    
    zone_pts_ref = np.array(zone_pts_ref, dtype=np.float32).reshape(-1, 1, 2)
    zone_homog = cv2.perspectiveTransform(zone_pts_ref, np.linalg.inv(H_total))

    for pt in zone_homog.reshape(-1,2).astype(int):
        cv2.circle(stabilized, tuple(pt), 5, (0,255,0), -1)
   
    cv2.imshow("Stabilized", stabilized)
    cv2.imshow("original", frame)
    if cv2.waitKey(10) == ord('q'):
        break
        

    prev_gray = gray
    kp1, des1 = kp2, des2
    
    print("frame number"+ f"{DebugCounter}")
    DebugCounter+=1
    
    
cap.release()
cv2.destroyAllWindows()
