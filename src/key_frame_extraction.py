
import cv2
import numpy as np
import os


try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')


# Playing video from file:
cap = cv2.VideoCapture('Video_Tests\A_Tilt_Side_to_Side.mp4')

currentFrame = 0

while cap.isOpened():
    #Establishing filename with the sequential number of frame
    filename = ('./data/frame') + str(currentFrame) + '.jpeg'
    #retval, image |grabs, decodes, and returns for next frame
    retrieved, frame = cap.read()
    #if next image was retrieved 
    if retrieved:
        cv2.imwrite(filename, frame)
        currentFrame += 3 #int determines how many frames to skip by
        cap.set(1, currentFrame)
    else:
        cap.release()
        break
