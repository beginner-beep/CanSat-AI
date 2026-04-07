import multiprocessing as mp
import time
import threading
import os
import cv2
import numpy as np
from gpsthread import gpsthread
from bmethread import bmethread
from antennathread import antennathread
from processqueue import mpq

#used for presentations as this script shows example screenrecordings we got from google maps
def manager(name, mpq):
    print(f"[{name}] Starting")

    stop_event = threading.Event()

    threads = [
        threading.Thread(target=gpsthread, args=(stop_event,), daemon=True),
        threading.Thread(target=bmethread, args=(stop_event,), daemon=True),
        threading.Thread(target=antennathread, args=(stop_event, mpq), daemon=True),
    ]

    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()


def ComputerVisionVideo(name, mpq, video_folder="TestVideos"):
    # HSV filter ranges (copied from original)
    lower_green = np.array([0, 0, 99])
    upper_green = np.array([255, 255, 255])
    lower_purple = np.array([113, 0, 0])
    upper_purple = np.array([179, 255, 255])

    def ApplyFilters(image):
        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        greenmask = cv2.inRange(hsvImage, lower_green, upper_green)
        purplemask = cv2.inRange(hsvImage, lower_purple, upper_purple)

        resgreen = cv2.bitwise_and(image, image, mask=greenmask)
        respurple = cv2.bitwise_and(image, image, mask=purplemask)

        finalRes = cv2.bitwise_or(resgreen, respurple)
        finalRes = cv2.cvtColor(finalRes, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv2.erode(finalRes, kernel, iterations=5)
        kernelClosing = np.ones((8, 8), np.uint8)

        ret, thresh = cv2.threshold(erosion, 10, 10, 10, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelClosing, iterations=3)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 500  # ignore tiny noise
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
        image_copy = thresh.copy()
        contours = filtered_contours[:3]  # keep only largest 3
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        scale = 0.5
        image_copy = cv2.resize(image_copy, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return image, image_copy

    video_paths = []
    folder_path = os.path.join(os.getcwd(), video_folder)
    if not os.path.isdir(folder_path):
        print(f"Video folder not found: {folder_path}")
        return

    for fname in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, fname)
    
        if not os.path.isfile(path) or fname.startswith("."):
            continue
    
        if not (fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))):
            continue
        video_paths.append(path)

    if not video_paths:
        print(f"No video files found in {folder_path}")
        return

    vid_index = 0
    try:
        while True:
            cap = cv2.VideoCapture(video_paths[vid_index])
            print(f"[ComputerVisionVideo] Playing: {video_paths[vid_index]}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image, image_copy = ApplyFilters(frame)

                cv2.imshow('frame', image_copy)
                cv2.imshow('erosion', image)

                try:
                    mpq.put_nowait("hello")
                except Exception:
                    pass

                if cv2.waitKey(10) == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cap.release()
            vid_index = (vid_index + 1) % len(video_paths)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        return


def main():
    print("Starting (test videos)")
    computer_vision_process = mp.Process(target=ComputerVisionVideo, args=("computer vision", mpq))
    data_process = mp.Process(target=manager, args=("other", mpq))

    computer_vision_process.start()
    data_process.start()

    computer_vision_process.join()
    data_process.join()


if __name__ == "__main__":
    main()
