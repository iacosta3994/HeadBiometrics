
import cv2
import numpy as np
import os


#Change in future to allow manipulation from __main__
cap_origin_video, saved_frames_location,  frames_to_skip  = ('Video_Tests\A_Tilt_Side_to_Side.mp4'), ('./data/frame') , 10

#function splits incoming mp4 and sends frames to saved_frames_locations
def split_frames(cap_origin_video, saved_frames_location, frames_to_skip):
    try:
        if not os.path.exists('data'):
            os.makedirs('data') #hosts images made in process
    except OSError:
        print ('Error: Creating directory of data')


    # Assigning video from file to cap:
    cap = cv2.VideoCapture(cap_origin_video)

    current_frame_idx = 0 #starting with the first scene

    while cap.isOpened():
        #Establishing filename with the sequential number of frame
        filename = saved_frames_location + str(current_frame_idx) + '.jpeg'

        #retval, image |grabs, decodes, and returns for next frame
        retrieved, frame = cap.read()

        #True if an image was retrieved
        if retrieved:
            cv2.imwrite(filename, frame)
            current_frame_idx += frames_to_skip # determines how many frames to skip by
            cap.set(1, current_frame_idx) #sets (propId, value) allows to skip to later frame in cap
        else: #No image input
            cap.release() #deallocates memory and clears capture pointer
            break

split_frames(cap_origin_video, saved_frames_location, frames_to_skip)
