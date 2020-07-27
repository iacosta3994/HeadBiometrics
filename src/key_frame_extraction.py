import cv2
import numpy as np
import os

global final_frame_idx #setting as global to allow update in function and able to return for parent function A_quantify

def split_frames(cap_origin_video, saved_frames_location, frames_to_skip): #function splits incoming mp4 and sends frames to saved_frames_locations

    try:
        if not os.path.exists('data'):
            os.makedirs('data') #hosts images made in process
    except OSError:
        print ('Error: Creating directory of data')

    cap = cv2.VideoCapture(cap_origin_video) # Assigning video from file to cap
    current_frame_idx = 0 #starting with the first scene
    while cap.isOpened():
        #Establishing filename with the sequential number of frame
        filename = saved_frames_location + str(current_frame_idx) + '.jpeg'

        #retval, image |grabs, decodes, and returns for next frame
        retrieved, frame = cap.read()

        #True if an image was retrieved
        if retrieved:
            cv2.imwrite(filename, frame)
            final_frame_idx = current_frame_idx #final frame idx to the value of current frame idx before returning final_frame_idx
            current_frame_idx += frames_to_skip #determines how many frames to skip by
            cap.set(1, current_frame_idx)       #set(propId, value) allows to skip to later frame in cap
        else: #No image input
            cap.release() #deallocates memory and clears capture pointer
            return(final_frame_idx) #return value will be used to determine stopping frame in A&B_quantify
